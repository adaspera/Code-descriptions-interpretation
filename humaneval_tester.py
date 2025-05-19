import datetime
import os
import json
import re
from vllm import LLM, SamplingParams
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from tqdm import tqdm
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
max_attempts = 3
results_file = f"humaneval_{model_name.replace('/', '-')}_{max_attempts}attempts.json"

sampling_params = SamplingParams(
    temperature=0.2,
    top_p=0.9,
    max_tokens=512,
)

def extract_code_from_markdown(text: str) -> str:
    pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

def test_with_inputs(generated_code: str, problem: dict) -> tuple[bool, str]:
    namespace = {}
    try:
        exec(generated_code, namespace)

        func_name = problem["entry_point"]
        func = namespace[func_name]

        prompt_lines = problem["prompt"].split('\n')
        import_lines = [line for line in prompt_lines if line.startswith(('import ', 'from '))]
        def_line = next(line for line in prompt_lines if line.startswith(f"def {func_name}"))
        
        complete_canonical = '\n'.join(import_lines) + '\n\n' + f"{def_line}\n{problem['canonical_solution']}"
        
        canonical_namespace = {}
        try:
            exec(complete_canonical, canonical_namespace)
            canonical_func = canonical_namespace[func_name]
        except Exception as e:
            return False, (f"Canonical solution error:\n{str(e)}\n\n")
        
        all_inputs = problem["base_input"]
        failures = []
        for inputs in all_inputs:
            try:
                generated_output = func(*inputs)
                expected_output = canonical_func(*inputs)
                
                if generated_output != expected_output:
                    failures.append(
                        f"Input: {inputs}\n"
                        f"Expected: {expected_output}\n"
                        f"Got: {generated_output}"
                    )
            except Exception as e:
                failures.append(f"Input: {inputs}\nError: {str(e)}")
        
        if failures:
            return False, "Base input test failures:\n" + "\n\n".join(failures)
        
        return True, None
        
    except Exception as e:
        return False, f"Error during testing: {str(e)}"
    

def iterative_code_tester(llm, tokenizer, problem, max_attempts=1):
    initial_prompt = (
        "Please complete the following Python function. "
        "Write the solution inside a markdown code block (```python) and include only the code. "
        "Don't include any explanations or text outside the code block.\n\n"
        f"{problem['prompt']}"
    )
    messages = [
        {"role": "system", "content": "You are a concise AI assistant. Provide only the requested code without explanations or step-by-step reasoning."},
        {"role": "user", "content": initial_prompt}
    ]
    
    attempts = 0
    history = []
    
    while attempts < max_attempts:
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = llm.generate([formatted_input], sampling_params)
        raw_output = outputs[0].outputs[0].text
        generated_code = extract_code_from_markdown(raw_output)
        
        passed, errors = test_with_inputs(generated_code, problem)
        
        if passed:
            return True, generated_code, history

        history.append({
            'attempt': attempts,
            'raw_output': raw_output,
            'extracted_code': generated_code,
            'errors': errors
        })

        messages.extend([
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": f"""
The code you provided failed with these errors:
{errors}

Please rewrite the function, fixing all mentioned issues.
"""}
        ])
        
        attempts += 1
    
    return False, generated_code, history


def load_existing_results():
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    return None

def save_progress(results):
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

def run_all_tasks():
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=16000,
        enforce_eager=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
     
    problems = get_human_eval_plus()
    
    existing_results = load_existing_results()
    
    if existing_results:
        evaluation_results = existing_results
        completed_tasks = {task["task_id"] for task in evaluation_results["tasks"]}
    else:
        evaluation_results = {
            "model": model_name,
            "start_time": datetime.datetime.now().isoformat(),
            "tasks": [],
            "stats": {
                "total_tasks": len(problems),
                "solved": 0,
                "failed": 0,
                "success_rate": 0.0
            }
        }
        completed_tasks = set()
    
    for task_id, problem in problems.items():
        
        if task_id in completed_tasks:
            print(f"Task {task_id} already completed, skipping...")
            continue
            
        print(f"Processing task {task_id}...")
        success, final_code, task_results = iterative_code_tester(llm, tokenizer, problem, max_attempts)
        print(f"Task {task_id}: {'Success' if success else 'Failed'}")
        
        evaluation_results["tasks"].append({
            "task_id": task_id,
            "success": success,
            "attempts": len(task_results),
            "final_code": final_code if success else None,
            "history": task_results
        })
        
        if success:
            evaluation_results["stats"]["solved"] += 1
        else:
            evaluation_results["stats"]["failed"] += 1
        
        evaluation_results["stats"]["success_rate"] = (
            evaluation_results["stats"]["solved"] / evaluation_results["stats"]["total_tasks"]
        )
        
        save_progress(evaluation_results)
    
    evaluation_results["end_time"] = datetime.datetime.now().isoformat()
    save_progress(evaluation_results)
    
    return evaluation_results, results_file

if __name__ == "__main__":
    results, filename = run_all_tasks()

    print(f"\nEvaluation completed. Results saved to {filename}")
    print(f"Solved: {results['stats']['solved']}/{results['stats']['total_tasks']}")
    print(f"Success rate: {results['stats']['success_rate']:.2%}")