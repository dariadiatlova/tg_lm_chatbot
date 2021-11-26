import pandas as pd
from typing import List, Union

from data_preprocessing.tg_json_reader import DataRetriever


def create_conversations(df: pd.DataFrame, min_dialoge_len: int = 3, max_dialoge_len: int = 500,
                         return_str: bool = True) -> Union[str, List[str]]:
    conversations = []
    idx_counter = 0

    while idx_counter < df.shape[0]:
        target_hour = df.time.loc[idx_counter].hour

        if isinstance(df.loc[idx_counter]["text"], str):
            message_block = ['']
        else:
            message_block = None

        for i in range(idx_counter, df.shape[0]):
            idx_counter += 1
            if df.time.loc[i].hour == target_hour:
                if isinstance(df.loc[i]["text"], str) and message_block:
                    if len(df.loc[i]["text"]) > 1:
                        message_block.extend([f'<{df.loc[i]["from"]}>' + " " + df.loc[i]["text"]])
                else:
                    continue
            else:
                if message_block:
                    if min_dialoge_len <= len(message_block) <= max_dialoge_len:
                        conversations.append(" ".join(message_block[1:]))
                break

    if return_str:
        return " ".join(conversations)

    return conversations


def get_conversations(path: str = "/Users/diat.lov/Downloads/Telegram Desktop/DataExport_2021-11-05/result.json"):
    data_retriever = DataRetriever(path)
    dictionary = data_retriever.get_message_history_by_name("Катя")
    df = data_retriever.get_pandas_df(dictionary)
    df = data_retriever.tg_specific_preprocessing(df)
    all_conversations = create_conversations(df, return_str=False)
    return all_conversations
