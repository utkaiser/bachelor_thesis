import os
import subprocess
from argparse import ArgumentParser
from jinja2 import Environment, FileSystemLoader

TEMPLATE_FILE = "kube.yaml.jinja2"

def script_args_to_string(dict_):
    string = ""
    for k, v in dict_.items():
        if isinstance(v, bool):
            if v:
                string += "\"--{}\", ".format(k)
        else:
            string += "\"--{}\", \"{}\", ".format(k, v)
    return string


def start_config(yaml_dict_list):
    script_path = os.path.dirname(os.path.abspath(__file__))
    template = Environment(loader=FileSystemLoader(script_path)).get_template(TEMPLATE_FILE)
    args_dict = {}

    number_of_runs = 3

    relevant_assets = [
                       #  'AAPL','ORCL',
                       #  'ACN','MSFT','IBM','CSCO','NVDA','ADBE','HPQ','INTC','TESS','ASYS','CTG',
                       # 'BELFB','AVNW','LYTS','JPM','BAC','V','PFE','MRK','JNJ','CAJ','NICE','TSM','SNE','UMC','CHKP',
                       # 'SILC','GILT','TSEM','LFC','SMFG','SHG','NOK','ASML','ERIC','SAP','TEL','LOGI','HSBC','ING',
                       #
                       # 'BCS','0992.HK','3888.HK','0763.HK','0939.HK','2318.HK','0998.HK','GC=F','SI=F','PL=F','HG=F',
                       # 'CL=F','HO=F','NG=F','RB=F','HE=F','LE=F','GF=F','DC=F','ZC=F','ZS=F','KC=F','KE=F','DIA','SPY',
                       # 'QQQ','EWJ','EWT','EWY','EZU','CAC.PA','EXS1.DE','EXXY.MI','DBC'
        "CL=F","EWY","DIA","JPM","DC=F","V","INTC","SMFG","EZU","NICE"
    ]

    for run in range(4,number_of_runs+4):
        for yaml_dict in yaml_dict_list:
            for asset in relevant_assets: #10*3*3 = 90
                args_dict["asset"] = asset
                args_dict["run"] = run
                args_dict["agent"] = yaml_dict["job_name"].replace("-","_")
                args_string = script_args_to_string(args_dict)
                output_text = template.render(log_dir="/results", args_str=args_string, **yaml_dict)
                command = "kubectl -n studkaiserl create -f -"
                p = subprocess.Popen(command.split(), stdin=subprocess.PIPE)
                p.communicate(output_text.encode())

if __name__ == '__main__':
    # Stuff that will be accessible by keys in the YAML configuration
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    debug = args.debug

    # Arguments which will be passed to the python script. Boolean flags will be automatically set to "--key" (if True)
    yaml_dict_list = [
        {
        "job_name": "q-learning-agent",
        "min_cpu": 1,
        "max_cpu": 1,
        "min_memory": 0.4,
        "max_memory": 5,
        "script_path": "/home/stud/kaiserl/bachelor_thesis/evaluate.py"},
        {"job_name": "duel-recurrent-q-learning-agent",
        "min_cpu": 6,
        "max_cpu": 6,
        "min_memory": 1,
        "max_memory": 22,
        "script_path": "/home/stud/kaiserl/bachelor_thesis/evaluate.py"},
        {"job_name": "actor-critic-agent",
        "min_cpu": 1,
        "max_cpu": 1,
        "min_memory": 0.5,
        "max_memory": 15,
        "script_path": "/home/stud/kaiserl/bachelor_thesis/evaluate.py"}
    ]
    start_config(yaml_dict_list)