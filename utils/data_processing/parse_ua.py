
from pandas import DataFrame, Series
from ua_parser import parse

ua_dtypes = {
        "String": "string",
        "Browser Name": "string",
        "Browser Version": "string",
        "Browser Minor": "string",
        "Browser Patch": "string",
        "Browser Patch Minor": "string",
        "OS Name": "string",
        "OS Version": "string",
        "OS Version Minor": "string",
        "OS Version Patch": "string",
        "OS Version Patch Minor": "string",
        "Device Brand": "string",
        "Device Model": "string",
        "Device Type": "string"
    }

ua_info_list = [name for name in ua_dtypes.keys()]

def ua_parser(ua_string):
    """
    Parse a user agent string and return a dictionary of information.

    :param ua_string: Text that describes the user agent
    :type ua_string: str
    :return: Dictionary of user agent information
    :rtype: dict
    """

    ua_data = parse(ua_string)
    ua_info = {"String": ua_data.string if ua_data.string is not None else None}
    if ua_data is not None:
        if ua_data.user_agent is not None:
            ua_info.update({
                "Browser Name": ua_data.user_agent.family,
                "Browser Version": ua_data.user_agent.major,
                "Browser Minor": ua_data.user_agent.minor,
                "Browser Patch": ua_data.user_agent.patch,
                "Browser Patch Minor" : ua_data.user_agent.patch_minor,
            })
        else:
            ua_info.update({
                "Browser Name": None,
                "Browser Version": None,
                "Browser Minor": None,
                "Browser Patch": None,
                "Browser Patch Minor" : None,
            })
        if ua_data.os is not None:
            ua_info.update({
                "OS Name": ua_data.os.family,
                "OS Version": ua_data.os.major,
                "OS Version Minor": ua_data.os.minor,
                "OS Version Patch": ua_data.os.patch,
                "OS Version Patch Minor": ua_data.os.patch_minor,
            })
        else:
            ua_info.update({
                "OS Name": None,
                "OS Version": None,
                "OS Version Minor": None,
                "OS Version Patch": None,
                "OS Version Patch Minor": None,
            })
        if ua_data.device is not None:
            ua_info.update({
                "Device Brand": ua_data.device.brand,
                "Device Model": ua_data.device.model,
                "Device Type": ua_data.device.family,
            })
        else:
            ua_info.update({
                "Device Brand": None,
                "Device Model": None,
                "Device Type": None
            })
    return ua_info

def list_ua_parser(ua_string: str) -> list:
    """
    Parse user agent string and return information as a list.
    
    :param ua_string: Text that describes the user agent
    :type ua_string: str
    :return: List of user agent information
    :rtype: list
    """

    ua_info = ua_parser(ua_string)
    return [ua_info[key] for key in ua_info_list]

def df_ua_parser(serie: Series) -> DataFrame:
    """
    Parse a DataFrame column containing user agent strings and return a DataFrame with parsed information.

    :param df: DataFrame containing user agent strings
    :type df: pandas.DataFrame
    :param column: Name of the column containing user agent strings
    :type column: str
    :return: DataFrame with parsed user agent information
    :rtype: pandas.DataFrame
    """

    ua_info_df = serie.apply(lambda x: Series(list_ua_parser(x), ))
    ua_info_df.columns = ua_info_list
    ua_info_df = ua_info_df.astype(ua_dtypes)

    return ua_info_df