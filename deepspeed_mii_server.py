import os
import mii
from time import perf_counter
import numpy as np


def measure_latency(pipe, prompt):
    latencies = []
    for _ in range(2):
        _ =  pipe.query(prompt)
    # Timed run
    for _ in range(10):
        start_time = perf_counter()
        _ = pipe.query(prompt,   
                 num_inference_steps=25,
                guidance_scale=7.5,
                num_images_per_prompt=1,
                )
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_s = np.mean(latencies)
    time_std_s = np.std(latencies)
    time_p95_s = np.percentile(latencies,95)
    return f"P95 latency (seconds) - {time_p95_s:.2f}; Average latency (seconds) - {time_avg_s:.2f} +\- {time_std_s:.2f};", time_p95_s


save_path = "."
deploy_name = "sd_deploy"

# Deploy Stable Diffusion w. MII
mii_config = {"dtype": "fp16",}
mii.deploy(task='text-to-image',
           model="runwayml/stable-diffusion-v1-5",
           deployment_name=deploy_name,
           mii_config=mii_config)

# Example usage of MII deployment
pipe = mii.mii_query_handle(deploy_name)
prompts = {"query": "a photo of an astronaut riding a horse on mars"}
results = pipe.query(prompts,   
                 num_inference_steps=25,
                guidance_scale=7.5,
                num_images_per_prompt=1,
                )
for idx, img in enumerate(results.images):
    img.save(os.path.join(save_path, f"mii-img{idx}.png"))

# Evaluate performance of MII
prompt = {"query": "a photo of an astronaut riding a horse on mars"}
results = measure_latency(pipe, prompt)

print(f"pipeline: {results[0]}")

# Tear down the persistent deployment
mii.terminate(deploy_name)
