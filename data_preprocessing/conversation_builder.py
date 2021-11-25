import pandas as pd


def create_conversations(df: pd.DataFrame, min_dialoge_len: int = 3, max_dialoge_len: int = 500):
    conversations = []
    idx_counter = 0

    while idx_counter < df.shape[0]:
        target_hour = df.time.loc[idx_counter].hour

        if isinstance(df.loc[idx_counter]["text"], str):
            message_block = [df.loc[idx_counter]["from"] + " " + df.loc[idx_counter]["text"]]
        else:
            message_block = None

        for i in range(idx_counter, df.shape[0]):
            idx_counter += 1
            if df.time.loc[i].hour == target_hour:
                if isinstance(df.loc[i]["text"], str) and message_block:
                    message_block.extend([df.loc[i]["from"] + " " + df.loc[i]["text"]])
                else:
                    continue
            else:
                if message_block:
                    if min_dialoge_len <= len(message_block) <= max_dialoge_len:
                        conversations.append(message_block)
                break
    return conversations
