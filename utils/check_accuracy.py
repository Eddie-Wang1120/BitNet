import os
import re
import argparse
import platform

# all weights irrelevant to ternary weights was set to fp32
# if interested in float32 output, please run
# ./build/bin/Release/llama-cli.exe -m ./models/bitnet_b1_58-large/ggml-model-f32.gguf -b 1 -t 4 -n 100 --seed 0
# with the same -p and --seed, i2_s/tl1/tl2/f32 shows exactly the same outputs.

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

TL_NAME = {
    "arm64": "tl1",
    "x86_64": "tl2"
}

def system_info():
    return platform.system(), ARCH_ALIAS[platform.machine()]

def file_location(arch, file):
    if arch == "x86_64":
        return os.path.join(".\\build\\bin\\Release", "{}.exe".format(file))
    elif arch == "arm64":
        return os.path.join("./build/bin/", file)

def setup_gguf():
    _, arch = system_info()
    tl_name = TL_NAME[arch]
    os.system("python setup_env.py -md ./models/bitnet_b1_58-large -q {} -fa".format(tl_name))
    os.system("python utils/convert-hf-to-gguf-bitnet.py ./models/bitnet_b1_58-large --outtype f32")
    os.system("{} --token-embedding-type f32 ./models/bitnet_b1_58-large/ggml-model-f32.gguf ./models/bitnet_b1_58-large/ggml-model-i2_s.gguf I2_S 1".format(file_location(arch, "llama-quantize")))
    os.system("{} --token-embedding-type f32 ./models/bitnet_b1_58-large/ggml-model-f32.gguf ./models/bitnet_b1_58-large/ggml-model-tq20.gguf TQ2_0".format(file_location(arch, "llama-quantize")))
    os.system("{} --token-embedding-type f32 ./models/bitnet_b1_58-large/ggml-model-f32.gguf ./models/bitnet_b1_58-large/ggml-model-tq10.gguf TQ1_0".format(file_location(arch, "llama-quantize")))

def setup_data():
    prompt_list = []
    with open(args.prompt_file, "r", encoding='utf-8', errors='ignore') as file:
        idx = 0
        for line in file:
            if idx >= args.data_num:
                break
            if line == '\n':
                continue
            idx = idx + 1
            line = line.replace("\n", "")
            prompt_list.append(line)

    return prompt_list

def generate_data(prompt_list):
    _, arch = system_info()
    txts = ["prompt.txt",
            "generate_result.txt",
            "{}.txt".format(args.kernel1),
            "{}.txt".format(args.kernel2)]

    print([os.path.join("results", txt) for txt in txts])

    txts = [os.path.join("results", txt) for txt in txts]
    for txt in txts:
        try:
            os.remove(txt)
        except:
            print("no {}".format(txt))

    if not os.path.exists("results"):
        os.makedirs("results")
    with open(os.path.join("results", "prompt.txt"), "a", encoding='utf-8') as file:
        for prompt in prompt_list:
            file.write(prompt)
            file.write("\n")

    for prompt in prompt_list:
        prompt = str(prompt).replace("\\", "\\\\")
        prompt = str(prompt).replace("\"", "\\\"")
        system_code = "{0} -m ./models/bitnet_b1_58-large/ggml-model-{1}.gguf \
            -b 1 -t {5} -n {4} -ngl 0 --seed {3} -p \"{2}\"" \
            .format(file_location(arch, "llama-cli") ,args.kernel1, prompt, str(args.seed)
                    , str(args.token_num), str(args.thread1))
        os.system(system_code)
    os.rename(os.path.join("results", "generate_result.txt"),
              os.path.join("results", "{}.txt".format(args.kernel1)))

    for prompt in prompt_list:
        prompt = str(prompt).replace("\\", "\\\\")
        prompt = str(prompt).replace("\"", "\\\"")
        system_code = "{0} -m ./models/bitnet_b1_58-large/ggml-model-{1}.gguf \
            -b 1 -t {5} -n {4} -ngl 0 --seed {3} -p \"{2}\"" \
            .format(file_location(arch, "llama-cli"), args.kernel2, prompt, str(args.seed)
                    , str(args.token_num), str(args.thread2))
        os.system(system_code)
    os.rename(os.path.join("results", "generate_result.txt"),
              os.path.join("results", "{}.txt".format(args.kernel2)))

def count_results(prompt_list):
    k1_result = []
    with open(os.path.join("results", "{}.txt".format(args.kernel1)),
              "r", encoding='utf-8', errors='ignore') as file:
        for line in file:
            if line == '\n':
                continue
            line = "<#>" + line[4:]
            tokens = line.split("<#>")[1:]
            tokens = [token.strip() for token in tokens]
            k1_result.append(tokens)

    k2_result = []
    with open(os.path.join("results", "{}.txt".format(args.kernel2)),
              "r", encoding='utf-8', errors='ignore') as file:
        for line in file:
            if line == '\n':
                continue
            line = "<#>" + line[4:]
            tokens = line.split("<#>")[1:]
            tokens = [token.strip() for token in tokens]
            k2_result.append(tokens)

    assert(len(k1_result) == len(k2_result))

    total_test = len(k1_result)
    wrong_test = 0
    right_test = 0
    wrong_prompt = []
    right_prompt = []

    for i in range(len(k1_result)):
        ptr = 0
        wrong_flag = False
        if len(k1_result[i]) != len(k2_result[i]):
            wrong_flag = True
        while ptr < len(k1_result[i]) and ptr < len(k2_result[i]) and not wrong_flag:
            if k1_result[i][ptr] != k2_result[i][ptr]:
                wrong_flag = True
            if not wrong_flag:
                ptr = ptr + 1
        if wrong_flag:
            wrong_test = wrong_test + 1
            wrong_prompt.append(prompt_list[i])
        else:
            right_test = right_test + 1
            right_prompt.append(prompt_list[i])

    assert(wrong_test + right_test == total_test)

    with open(os.path.join("results", "wrong_prompt.txt"), "w", encoding='utf-8') as file:
        for prompt in wrong_prompt:
            file.write(prompt)
            file.write("\n")
    with open(os.path.join("results", "right_prompt.txt"), "w", encoding='utf-8') as file:
        for prompt in right_prompt:
            file.write(prompt)
            file.write("\n")

    print("total: {}".format(str(total_test)))
    print("right: {}".format(str(right_test)))
    print("wrong: {}".format(str(wrong_test)))
    print("accuracy: {}".format(float(right_test) / float(total_test)))

def fresh_env():
    dir_path = "./models/bitnet_b1_58-large"
    pattern = re.compile(r'ggml.*\.gguf')
    files = [file for file in os.listdir(dir_path) if pattern.match(file)]
    for file in files:
        os.remove("".join([dir_path, '/', file]))

def main():
    fresh_env()
    setup_gguf()
    prompt_list = setup_data()
    assert(len(prompt_list) == args.data_num)
    generate_data(prompt_list)
    count_results(prompt_list)

def parse_args():
    _, arch = system_info()
    parser = argparse.ArgumentParser(description='compare outputs')
    parser.add_argument("--kernel1", "-k1", type=str, choices=SUPPORTED_QUANT_TYPES[arch])
    parser.add_argument("--kernel2", "-k2", type=str, choices=SUPPORTED_QUANT_TYPES[arch])
    parser.add_argument("--thread1", "-t1", type=int, default=4)
    parser.add_argument("--thread2", "-t2", type=int, default=4)
    parser.add_argument("--data-num", "-n", type=int, default=5, help="compare data num")
    parser.add_argument("--model", "-m", type=str, choices=["bitnet_b1_58-large"], help="right now only suits for bitnet_b1_58-large")
    parser.add_argument("--prompt-file", "-f", type=str, default="./prompt/WildChat-1M.txt", help="prompt file")
    parser.add_argument("--seed", "-s", type=int, default=0, help="seed")
    parser.add_argument("--token-num", "-t", type=int, default=100, help="number of generate tokens")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main()
