def extract_info_from_file(file_path):
    info_list = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # print("lines: ",lines)
        i = 0
        while i < len(lines):
            if lines[i].startswith("       ------- seed:"):
                seed = int(lines[i].split(":")[1].strip())
                record_line = lines[i + 9]
                score_line = lines[i + 6]
                time_line = lines[i + 4]
                # print(f"line time {time_line}, record {record_line}, score {score_line}")
                record = record_line.split(":")[1].strip()
                score = float(score_line.split(":")[1].strip())
                time = float(time_line.split(":")[1].strip().split()[0])
                info_list.append((seed, record, score, time))
                i += 7  # Chuyển tới random seed tiếp theo
            else:
                i += 1
    return info_list

infoList = extract_info_from_file(r'C:\Users\thuyl\OneDrive\My documents\AI\CS106_Artificial-Intelligence\BT4_22520766\multiagent\results\ExpectimaxAgent_betterEvalfc.txt')
for in4 in infoList:
    for inf in in4:
        print(inf, sep='\n')
    print('\n')