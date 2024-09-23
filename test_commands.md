### cuda requirements
`python -c 'from transformers import AutoModel; from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; model = AutoModelForSequenceClassification.from_pretrained("Salesforce/codet5p-220m"); estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=3, num_nodes=1)'`

[2024-09-10 12:55:50,347] [INFO] [real_accelerator.py:203:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Warning: The default cache directory for DeepSpeed Triton autotune, /vol/fob-vol5/mi14/kohlmane/.triton/autotune, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
 [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
 [WARNING]  async_io: please install the libaio-devel package with yum
 [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.3
 [WARNING]  using untested triton version (2.3.1), only 1.0.0 is known to be compatible
Some weights of T5ForSequenceClassification were not initialized from the model checkpoint at Salesforce/codet5p-220m and are newly initialized: ['classification_head.dense.bias', 'classification_head.dense.weight', 'classification_head.out_proj.bias', 'classification_head.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 3 GPUs per node.
SW: Model with 223M total params, 24M largest layer params.
  per CPU  |  per GPU |   Options
    5.62GB |   0.09GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
    5.62GB |   0.09GB | offload_param=OffloadDeviceEnum.cpu, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
    5.00GB |   0.23GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=1
    5.00GB |   0.23GB | offload_param=none, offload_optimizer=OffloadDeviceEnum.cpu, zero_init=0
    0.41GB |   1.34GB | offload_param=none, offload_optimizer=none, zero_init=1
    3.75GB |   1.34GB | offload_param=none, offload_optimizer=none, zero_init=0

### Slurm
On a GPU the job is run with Slurm, so the arguments are specified in the script `pytorch_gpu.sh`.   
To start a job on a GPU use the command `sbatch pytorch_gpu.sh`. 


### command finetuning
`nice -n 19 python3 finetuning.py --vuln_type=sql  --cache_data=../cache_data/sql --save_dir=../saved_models/sql --data_num=20000`

### command gpu_fine-tuning
nice -n 19 python3 gpu_fine_tuning.py --vuln_type=xsrf  --cache_data=../cache_data/xsrf --save_dir=../saved_models/xsrf --per_device_train_batch_size=1 --per_device_eval_batch_size=1


`nice -n 19 python3 gpu_fine_tuning.py --vuln_type=open_redirect  --cache_data=../cache_data/open_redirect --save_dir=../saved_models/gpu_open_redirect_770 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --epochs=15 --data_num=20000 --load=Salesforce/codet5p-770m`

## hyperparameter search
`nice -n 19 python3 hyperparameter_search.py --vuln_type=xss --cache_data=../cache_data/xss --save_dir=../hyperparameter_search/hps_xss_full --n_trials=5`

Smaller `--data-num` results in `TypeError: object of type 'NoneType' has no len()`


### command few-shot-prompting
`nice -n 19 python few_shot_prompting.py --vuln-type=xss --cache-data=../cache_data/xss`




### opensuse auf gruenau2
cat /etc/os-release to display OpenSUSE/SUSE Linux version
NAME="openSUSE Leap"
VERSION="15.5"
ID="opensuse-leap"
ID_LIKE="suse opensuse"
VERSION_ID="15.5"
PRETTY_NAME="openSUSE Leap 15.5"
ANSI_COLOR="0;32"
CPE_NAME="cpe:/o:opensuse:leap:15.5"
BUG_REPORT_URL="https://bugs.opensuse.org"
HOME_URL="https://www.opensuse.org/"
DOCUMENTATION_URL="https://en.opensuse.org/Portal:Leap"
LOGO="distributor-logo-Leap"
cat: to: Datei oder Verzeichnis nicht gefunden
cat: display: Datei oder Verzeichnis nicht gefunden
cat: OpenSUSE/SUSE: Datei oder Verzeichnis nicht gefunden
cat: Linux: Datei oder Verzeichnis nicht gefunden
cat: version: Datei oder Verzeichnis nicht gefunden