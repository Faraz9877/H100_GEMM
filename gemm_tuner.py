import itertools
import subprocess
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from jinja2 import Template
import time

class GemmTuner:
    def __init__(self, template_file: str, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define parameter search space
        self.param_space = {
            'CLUSTER_M': [1, 2, 4, 8, 16],
            'CLUSTER_N': [1],
            'BLOCKM': [64, 128, 256],
            'BLOCKN': [128],
            'BLOCKK': [64],
            'STAGES': [3, 5, 7, 9, 11]
        }
        
        # Load template
        with open(template_file, 'r') as f:
            self.template = Template(f.read())

    def generate_code(self, params: Dict[str, int]) -> str:
        """Generate code with given parameters"""
        return self.template.render(**params)

    def benchmark_configuration(self, params: Dict[str, int]) -> Tuple[float, bool]:
        """Benchmark a single configuration. Returns (runtime, success)"""
        # Write the generated code to temp.cu
        with open('temp.cu', 'w') as f:
            f.write(self.generate_code(params))
        
        # If test file exists, remove it
        test_file = Path('test')
        if test_file.exists():
            test_file.unlink()
        
        # Compile the code
        compile_result = subprocess.run('nvcc -arch=sm_90a -lcuda -std=c++17 temp.cu -o test > /dev/null 2> /dev/null', 
                                      shell=True)
        
        if compile_result.returncode != 0:
            print(f"Compilation failed for parameters: {params}")
            return float('inf'), False
            
        # Run the benchmark
        try:
            result = subprocess.run(['./test', 'M', '5120', 'N', '5120', 'K', '4096'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Runtime error for parameters: {params}")
                return float('inf'), False
                
            # Extract runtime from output
            match = re.search(r"Average latency \(ms\) is ([\d.]+)", result.stdout)
            if match:
                runtime = float(match.group(1))
                return runtime, True
            print(f"Could not parse output for parameters: {params}")
            return float('inf'), False
            
        except subprocess.TimeoutExpired:
            print(f"Timeout for parameters: {params}")
            return float('inf'), False

    def is_valid_configuration(self, params: Dict[str, int]) -> bool:
        """Check if a parameter configuration is valid"""
        # Check cluster size
        if params['CLUSTER_N'] > 1:
            return False
            
        return True

    def tune(self):
        """Run the tuning process and save results"""
        results = []
        best_runtime = float('inf')
        best_params = None
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(self.param_space.keys(), values)) 
                            for values in itertools.product(*self.param_space.values())]
        
        start_time = datetime.now()
        total_configs = len([p for p in param_combinations if self.is_valid_configuration(p)])
        tested_configs = 0
        
        for params in param_combinations:
            if not self.is_valid_configuration(params):
                continue
                
            tested_configs += 1
            print(f"\nTesting configuration {tested_configs}/{total_configs}:")
            print(params)
            
            runtime, success = self.benchmark_configuration(params)
            
            result = {
                'params': params,
                'runtime': runtime if success else None,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)

            if success:
                print(f"Runtime: {runtime:.4f} ms")
            
            if success and runtime < best_runtime:
                best_runtime = runtime
                best_params = params
                print(f"New best runtime: {best_runtime:.4f} ms")
                
                # Save the best configuration immediately
                best_config_file = self.output_dir / 'best_matmul_h100.cu'
                with open(best_config_file, 'w') as f:
                    f.write(self.generate_code(best_params))
                
            # Save intermediate results
            self.save_results(results, best_params, best_runtime)
            
        return best_params, best_runtime, results

    def save_results(self, results: List[Dict], best_params: Dict[str, int], best_runtime: float):
        """Save results to disk"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = self.output_dir / f'tuning_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'results': results,
                'best_params': best_params,
                'best_runtime': best_runtime,
                'timestamp': timestamp
            }, f, indent=2)
            
        # Save summary
        summary_file = self.output_dir / 'tuning_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Tuning completed at: {timestamp}\n")
            f.write(f"Best runtime: {best_runtime:.4f} ms\n")
            if best_params:
                f.write("Best parameters:\n")
                for key, value in best_params.items():
                    f.write(f"{key}: {value}\n")
            
            f.write("\nAll successful configurations:\n")
            sorted_results = sorted(
                [r for r in results if r['success']], 
                key=lambda x: x['runtime']
            )
            for r in sorted_results:
                f.write(f"\nRuntime: {r['runtime']:.4f} ms\n")
                for k, v in r['params'].items():
                    f.write(f"{k}: {v}\n")

def main():
    tuner = GemmTuner('matmul_h100.cu.jinja', 'tuning_results')
    best_params, best_runtime, results = tuner.tune()
    
    if best_params:
        print("\nTuning completed!")
        print(f"Best runtime: {best_runtime:.4f} ms")
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"{key}: {value}")
        print("\nResults saved in tuning_results directory")
    else:
        print("\nNo valid configuration found!")

if __name__ == "__main__":
    main()