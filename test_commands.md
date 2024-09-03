### command finetuning
`nice -n 19 python3 finetuning.py --vuln_type=sql  --cache_data=../cache_data/sql --save_dir=../saved_models/sql --data_num=20000`

### command gpu_fine-tuning
nice -n 19 python3 gpu_fine_tuning.py --vuln_type=path_disclosure  --cache_data=../cache_data/path_disclosure --save_dir=../saved_models/path_disclosure_770 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --epochs=15 --data_num=20000 --load=Salesforce/codet5p-770m


`nice -n 19 python3 gpu_fine_tuning.py --vuln_type=open_redirect  --cache_data=../cache_data/open_redirect --save_dir=../saved_models/gpu_open_redirect_770 --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --epochs=15 --data_num=20000 --load=Salesforce/codet5p-770m`

## hyperparameter search
`nice -n 19 python3 hyperparameter_search.py --vuln_type=xss --data_num=20000 --n_trials=1 --cache_data=../cache_data/xss`

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