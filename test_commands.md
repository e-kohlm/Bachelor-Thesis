### command fine-tuning
`nice -n 19 python fine_tuning_tests.py --vuln-type=xss --cache-data=../cache_data/xss`

### command few-shot-prompting
`nice -n 19 python few_shot_prompting.py --vuln-type=xss --cache-data=../cache_data/xss`

### GPU estimate
`python -c 'from transformers import AutoModel;
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live;
model = AutoModel.from_pretrained("Salesforce/codet5p-770m");
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'`   

OUTPUT:   
`Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 737M total params, 32M largest layer params.
  per CPU  |  per GPU |   Options
   18.55GB |   0.12GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
   18.55GB |   0.12GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
   16.49GB |   1.50GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
   16.49GB |   1.50GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
    0.18GB |  12.49GB | offload_param=none, offload_optimizer=none, zero_init=1
    4.12GB |  12.49GB | offload_param=none, offload_optimizer=none, zero_init=0`