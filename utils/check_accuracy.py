import pandas as pd
import langdetect
import os
import argparse
import platform

# all weights irrelevant to ternary weights was set to fp32
# if interested in float32 output, please run
# ./build/bin/Release/llama-cli.exe -m ./models/bitnet_b1_58-large/ggml-model-f32.gguf -b 1 -t 4 -n 100 --seed 0
# with the same -p and --seed, i2_s/tl1/tl2/f32 shows exactly the same outputs.

def setup_data():
    os.system("python setup_env.py -md .\\models\\bitnet_b1_58-large -q tl2 -fa")
    os.system("python utils\\convert-hf-to-gguf-bitnet.py --outtype f32 .\\models\\bitnet_b1_58-large")
    os.system(".\\build\\bin\\Release\\llama-quantize.exe .\\models\\bitnet_b1_58-large\\ggml-model-f32.gguf .\\models\\bitnet_b1_58-large\\ggml-model-i2_s.gguf I2_S 1")
    os.system(".\\build\\bin\\Release\\llama-quantize.exe .\\models\\bitnet_b1_58-large\\ggml-model-f32.gguf .\\models\\bitnet_b1_58-large\\ggml-model-tq20.gguf TQ2_0")
    os.system(".\\build\\bin\\Release\\llama-quantize.exe .\\models\\bitnet_b1_58-large\\ggml-model-f32.gguf .\\models\\bitnet_b1_58-large\\ggml-model-tq10.gguf TQ1_0")

    data = pd.read_parquet(".\\WildChat-1M\\data\\train-00000-of-00014.parquet")
    data = data[data['country'] == 'United States'][data['language'] == 'English']
    data_list = data['conversation'].to_list()
    prompt_set = set()
    for arr in data_list:
        for data in arr:
            tokens = data['content']
            if "\n" not in tokens:
                prompt_set.add(tokens)

    prompt_list = []
    idx = 0
    for prompt in prompt_set:
        if idx >= args.data_num:
            break
        if len(prompt) > 100:
            continue
        try:
            if langdetect.detect(str(prompt)) == 'en':
                idx = idx + 1
                prompt_list.append(prompt)
        except:
            print(prompt)
            print("no feature")
    return prompt_list

def generate_data(prompt_list):
    try:
        os.remove("prompt.txt")
    except:
        print("no prompt.txt")
    try:
        os.remove("generate_result.txt")
    except:
        print("no generate_result.txt")
    try:
        os.remove("{}.txt".format(args.kernel1))
    except:
        print("no {}.txt".format(args.kernel1))
    try:
        os.remove("{}.txt".format(args.kernel2))
    except:
        print("no {}.txt".format(args.kernel2))

    with open("prompt.txt", "a", encoding='utf-8') as file:
        for prompt in prompt_list:
            file.write(prompt)
            file.write("\n")

    for prompt in prompt_list:
        system_code = ".\\build\\bin\\Release\\llama-cli.exe -m .\\models\\bitnet_b1_58-large\\ggml-model-{0}.gguf \
            -b 1 -t 4 -n 100 --seed 0 -p \"{1}\"".format(args.kernel1, prompt)
        os.system(system_code)
    os.rename("generate_result.txt", "{}.txt".format(args.kernel1))

    for prompt in prompt_list:
        system_code = ".\\build\\bin\\Release\\llama-cli.exe -m .\\models\\bitnet_b1_58-large\\ggml-model-{0}.gguf \
            -b 1 -t 4 -n 100 --seed 0 -p \"{1}\"".format(args.kernel2, prompt)
        os.system(system_code)
    os.rename("generate_result.txt", "{}.txt".format(args.kernel2))

def count_results(prompt_list):

    k1_result = []
    with open("{}.txt".format(args.kernel1), "r", encoding='utf-8', errors='ignore') as file:
        for line in file:
            line = "<#>" + line[4:]
            tokens = line.split("<#>")[1:]
            tokens = [token.strip() for token in tokens]
            k1_result.append(tokens)
    k1_result = k1_result[1:]

    k2_result = []
    with open("{}.txt".format(args.kernel2), "r", encoding='utf-8', errors='ignore') as file:
        for line in file:
            line = "<#>" + line[4:]
            tokens = line.split("<#>")[1:]
            tokens = [token.strip() for token in tokens]
            k2_result.append(tokens)
    k2_result = k2_result[1:]

    assert(len(k1_result) == len(k2_result))

    total_test = len(k1_result)
    wrong_test = 0
    right_test = 0
    wrong_prompt = []

    for i in range(len(k1_result)):
        ptr = 0
        wrong_flag = False
        if len(k1_result[i]) != len(k2_result[i]):
            wrong_test = wrong_test + 1
            wrong_flag = True
        while ptr < len(k1_result[i]) and ptr < len(k2_result[i]) and not wrong_flag:
            if k1_result[i][ptr] != k2_result[i][ptr]:
                wrong_test = wrong_test + 1
                wrong_flag = True
            ptr = ptr + 1
        if wrong_flag:
            wrong_prompt.append(prompt_list[i])
        if not wrong_flag:
            right_test = right_test + 1

    assert(wrong_test + right_test == total_test)

    print(wrong_prompt)

    print("total: {}".format(total_test))
    print("right: {}".format(right_test))
    print("accuracy: {}".format(float(right_test) / float(total_test)))

SUPPORTED_QUANT_TYPES = {
    "arm64": ["i2_s", "tl1", "tq10", "tq20", "f32"],
    "x86_64": ["i2_s", "tl2", "tq10", "tq20", "f32"]
}

ARCH_ALIAS = {
    "AMD64": "x86_64",
    "x86": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}

def system_info():
    return platform.system(), ARCH_ALIAS[platform.machine()]

def main():
    prompt_list = setup_data()
    generate_data(prompt_list)
    count_results(prompt_list)

def parse_args():
    _, arch = system_info()
    parser = argparse.ArgumentParser(description='compare outputs')
    parser.add_argument("--kernel1", "-k1", type=str, choices=SUPPORTED_QUANT_TYPES[arch])
    parser.add_argument("--kernel2", "-k2", type=str, choices=SUPPORTED_QUANT_TYPES[arch])
    parser.add_argument("--data_num", "-n", type=int, default=5, help="compare data num")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main()
