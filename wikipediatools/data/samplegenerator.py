from wikipediatools.data.features import FeaturesExtractor, FeatureSets
from wikipediatools.data.utils import get_page_content
import mwparserfromhell
from tqdm import tqdm
import pandas as pd
import random


class SampleGenerator():
    """
    A class that generates a sample dataset and extracts features from it.
    """

    def __init__(self, grading_dataset :pd.DataFrame, size_per_grading : int, random_state: int = 42, save_raw_data = False, include_c_class=True) -> None:
        """
        A class that generates a sample dataset and extracts features from it.

        Args:
            grading_dataset (pandas.DataFrame): The input DataFrame containing the gradings dataset.
            size_per_grading (int): The number of samples to be generated per grading.
            random_state (int, optional): The random state to be used for sampling. Defaults to 42.
            save_raw_data (bool, optional): Whether to save the raw data that contains the page content. Defaults to False.
            include_c_class (bool, optional): Whether to include C class in the sample dataset. Defaults to True.
        """
        self.df = grading_dataset
        self.random_state = random_state
        self.size_per_grading = size_per_grading
        self.save_raw_data = save_raw_data
        self.grades = ["Stub","Start","B","C","A","GA","FA"]
        if not include_c_class:
            self.grades.remove("C")

    def _generate_sample(self):
        """
        Generates a sample dataset that's evenly distrbuted.

        Returns:
            pandas.DataFrame: The generated sample dataset.
        """
        df = self.df.copy()
        df = df[df["Grade"].isin(self.grades)].dropna()
        df = df[df["RevID"] != "Missing"]
        try:
            sample = df.groupby("Grade").apply(lambda x: x.sample(self.size_per_grading, random_state=self.random_state)).reset_index(drop=True)
        except ValueError:
            print("Info: The sample size per grading is larger than the at least one grading, sampling with replacement")
            sample = df.groupby("Grade").apply(lambda x: x.sample(self.size_per_grading, random_state=self.random_state, replace=True)).reset_index(drop=True)
        sample["RevID"] = sample["RevID"].astype(int) 
        return sample

    def _generate_sample_dataset(self):
        """
        Retrieve the text for the sample dataset.

        Args:
            store (bool, optional): Whether to store the sample dataset as a CSV file. Defaults to True.
        
        Returns:
            pandas.DataFrame: The sample dataset with text.
        """
        # Generate the sample dataset
        sample = self._generate_sample()

        chunk = []
        temp = {"RevID": [], "Content": []}

        for i, revID in enumerate(tqdm(sample["RevID"].values, desc="Retrieving text")):
            chunk.append(str(revID))

            if len(chunk) == 50 or len(sample) - 1 == i:
                ids = "|".join(chunk)
                chunk = []

                # Retrieve the page content for the chunked RevIDs
                for rev_id, content in get_page_content(ids):
                    temp["RevID"].append(rev_id)
                    temp["Content"].append(content)

        df = pd.DataFrame(temp)
        df["RevID"] = df["RevID"].astype(int)
        df["Content"] = df["Content"].astype(str, errors="ignore")

        # Merge the sample dataset with the retrieved text on RevID
        merged_df = pd.merge(sample, df, "inner", on="RevID").drop_duplicates()

        # Filter out any rows with missing RevID
        merged_df = merged_df[df["RevID"] != "Missing"]

        if self.save_raw_data:
            merged_df.to_csv("sample_dataset.csv", index=False)

        return merged_df
    
    def _extract_items(self, extractor: FeaturesExtractor, title, grade, revid, code):
        """
        Extracts features from a single sample.
        
        Args:
            extractor (FeaturesExtractor): The feature extractor object.
            title (str): The title of the sample.
            grade (str): The grade of the sample.
            revid (int): The revision ID of the sample.
            code (mwparserfromhell.wikicode.Wikicode): The parsed code of the sample.

        Returns:
            dict: The extracted features.
        """
        try:
            items = extractor.extract_feature_sets(title, revid, code).items()
        except Exception:
            generator = SampleGenerator(self.df, 1, random.randint(100, 1000))
            temp_df = generator._generate_sample_dataset(False)
            temp_df = temp_df[temp_df["Grade"] == grade].sample(1)
            old_title = title
            grade, title, content, revid = temp_df[["Grade", "Title", "Content", "RevID"]].values[0]
            code = mwparserfromhell.parse(content)
            items = extractor.extract_feature_sets(title, revid, code).items()
            print(f"""Warning: Feature extraction failed,
            generating a new sample, the sample: {old_title} has been replaced with {title}""")

        return items, grade, title, revid

    def get_features(self, save_path, feature_sets=FeatureSets.ALL, df = None):
        """
        Extracts features from the sample dataset and saves it as a CSV file.

        Args:
            save_path (str): The path to save the CSV file.
            feature_sets (FeatureSets, optional): The feature sets to be extracted. Defaults to FeatureSets.ALL.
            df (pandas.DataFrame, optional): custom dataframe to extract features from it 
                must include the following columns Title, Grade, Content, RevID.

        Returns:
            pandas.DataFrame: The extracted features dataframe.
        """
        if FeatureSets._CUSTOM == feature_sets or (isinstance(feature_sets, list) and FeatureSets._CUSTOM in feature_sets):
            raise ValueError("FeatureSets._CUSTOM is not supported")
        if FeatureSets._EMBEDDINGS == feature_sets or (isinstance(feature_sets, list) and FeatureSets._EMBEDDINGS in feature_sets):
            raise ValueError("FeatureSets._EMBEDDINGS is not supported")

        
        if df is None:
            df = self._generate_sample_dataset()
            
        extractor = FeaturesExtractor(feature_sets)
        data = {"Title":[],"Grade":[], "RevID":[]}
        index = 0

        for title, grade, content, revid in tqdm(df[["Title", "Grade", "Content", "RevID"]].values, desc="Extracting features"):
            code = mwparserfromhell.parse(content)

            # Skip if the code is a redirect or empty template
            if len(code.strip_code()) < 50:
                continue
            
            while True:
                try:
                    items, grade, title, revid  = self._extract_items(extractor, title, grade, revid, code)
                    break
                except Exception:
                    pass

            data["Grade"].append(grade)
            data["Title"].append(title)
            data["RevID"].append(revid)
            index += 1
            for k,v in items:
                try:
                    data[k].append(v)
                except KeyError:
                    data[k] = [v]
            if index % 100 == 0:
                self._create_checkpoint(save_path, data)

        self._create_checkpoint(save_path, data)
        return df

    def _create_checkpoint(self, path, data):
        df = pd.DataFrame(data)

        # Added Target
        df.loc[df["Grade"].isin(["FA","GA","A"]), "Target"] = "High"
        df.loc[df["Grade"].isin(["B","Start","Stub"]), "Target"] = "Low"

        df.to_csv(f"{path}.csv", index=False)


        

                
