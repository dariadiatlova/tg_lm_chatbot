import json
import pandas as pd
from typing import Union, List
from pathlib import Path
from loggers.get_logger import get_logger

path = "/Users/diat.lov/Downloads/Telegram Desktop/DataExport_2021-11-05/result.json"


class DataRetriever:
    def __init__(self, json_file_path: str):
        self.config = self.read_json(json_file_path)
        self.chats_data = self.config["chats"]["list"]
        self.logger = get_logger("DataRetriever")

    @staticmethod
    def read_json(path: str = path):
        return json.load(open(Path(path)))

    def file_structure(self, field: str = None):
        """
        Main dict fields are: 'about', 'personal_information', 'profile_pictures',
        'contacts', 'frequent_contacts', 'chats'.
        :param field: name for the dict field to get the keys
        :return: list[str], keys to access the fields inside the given
        """

        if field:
            return self.config[f"{field}"].keys()
        return self.config.keys()

    def get_message_history_by_name(self, name: str) -> Union[str, List[dict]]:
        for user_chat in self.chats_data:
            if "name" not in user_chat.keys():
                continue
            if user_chat["name"] == name:
                return user_chat["messages"]
        return "No chat found with the specified name :("

    def get_message_history_by_id(self, user_id: int) -> Union[str, List[dict]]:
        for user_chat in self.chats_data:
            if user_chat["id"] == user_id:
                return user_chat["messages"]
        return "No chat found with the specified name :("

    def get_top_n_frequent_contacts(self, n: int) -> str:
        contact_list_dictionary = self.config["frequent_contacts"]["list"]
        list_with_names = [personal_info["name"] for personal_info in contact_list_dictionary[:n]]
        return f"Recently you have been communicated a lot with: {list_with_names} :)"

    def get_pandas_df(self, data_list: List[dict], columns_to_extract: List[str] = None):
        """
        Method takes list with dicts and return DataFrame with the same fields.
        :param data_list:
        :return: pd.DataFrame
        """
        self.logger.info(f"The number of messages: {len(data_list)}")
        return pd.DataFrame(data_list)


# data_retriever = DataRetriever(path)
# dictinoary = data_retriever.get_message_history_by_name("Катя")
# print(data_retriever.get_pandas_df(dictinoary).columns)
