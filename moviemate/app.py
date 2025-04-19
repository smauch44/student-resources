# load import statements

def load_dataset(file_path, separation_type=None):
    """
    Load the dataset using pandas and inspect its structure.
    This load is for the purpose of returning a pandas.DataFrame which will 
    later get partitioned into test and train data.
    :param file_path: path to data file that will be loaded
    :param separation_type: way that the file is separated, tab (`\t`) or bar (`|`)
    :return: pandas.Dataframe of the dataset
    """
    pass


def partition_data(ratings_df, split: int = .8, partition_type='stratified', stratify_by='user_id'):
    """
    Split the data into training and testing sets using user-stratified sampling and temporal sampling.
    :param partition_type: partitioning strategy. stratified (Stratified Sampling), or temporal (Temporal Sampling).
    :return: A tuple containing:
        - train_df: pandas.DataFrame 
            training dataset
        - test_df: pandas.DataFrame 
            testing dataset
    """
    pass