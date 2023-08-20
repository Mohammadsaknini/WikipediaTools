from typing import Generator, Tuple
from tqdm import tqdm
import pandas as pd
import requests
import mwxml
import csv


def get_page_assessment(revid: str, headers: dict = {}) -> Generator[Tuple[str, str], None, None]:
    """
    Retrieves page assessments for a given revision ID(s) from the Wikipedia API.

    Parameters:
    - revid (str): A string containing one or multiple revision IDs separated by '|'.
    - headers (dict): An optional dictionary containing additional headers for the API request. If not provided, it defaults to an empty dictionary.

    Yields:
    - Tuple[str, str]: A tuple containing the page title and grade of the page assessment.

    Note:
    The function uses the Wikipedia API to fetch page assessments. It requires a valid User-Agent header to be set. You can pass additional headers using the `headers` parameter.

    Example:
    ```
    for rev_id, title, grade in get_page_assessment("12345|67890"):
        print(f"Revision ID: {rev_id}, Title: {title}, Grade: {grade}")
    ```
    """
    if not headers.get('Accept-Encoding'):
        headers['Accept-Encoding'] = 'gzip'

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "prop": "pageassessments",
        "revids": revid,
        "format": "json",
        "palimit": "500",
    }

    response = requests.get(url, params=params, headers=headers).json()

    # Extract the page IDs from the response
    pages_id = list(response["query"]["pages"].keys())
    pages = response["query"]["pages"]

    # Iterate over the page IDs and yield the title and grade
    for page_id in pages_id:
        try:
            title = pages[page_id]["title"]
            category = list(pages[page_id]["pageassessments"].keys())[0]
            grade = pages[page_id]["pageassessments"][category]["class"]
            yield title, grade
        except KeyError:
            pass

def get_page_content(revid: str, headers: dict = {}) -> Generator[Tuple[str, str], None, None]:
    """
    Retrieves the content of a Wikipedia page for a given revision ID from the Wikipedia API.

    Parameters:
    - revid (str): A string representing the revision ID.
    - headers (dict): An optional dictionary containing additional headers for the API request. If not provided, it defaults to an empty dictionary.

    Yields:
    - Tuple[str, str]: A tuple containing the revision ID and the content of the Wikipedia page.

    Note:
    The function uses the Wikipedia API to fetch page content. It requires a valid User-Agent header to be set. You can pass additional headers using the `headers` parameter.

    Example:
    ```
    for rev_id, content in get_page_content("12345"):
        print(f"Revision ID: {rev_id}, Content: {content}")
    ```
    """
    if not headers.get('Accept-Encoding'):
        headers['Accept-Encoding'] = 'gzip'

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "prop": "revisions",
        "revids": revid,
        "rvprop": "content|ids",
        "format": "json"
    }

    response = requests.get(url, params=params, headers=headers).json()

    # Extract the page ID from the response
    pages_id = list(response["query"]["pages"].keys())
    pages = response["query"]["pages"]

    # Iterate over the page IDs and yield the revision ID and content
    for page_id in pages_id:
        try:
            revision = pages[page_id]["revisions"][0]
            content = revision["*"]
            rev_id = revision["revid"]
            yield rev_id, content
        except KeyError:
            pass

def fetch_grades(dump: mwxml.Dump, path :str, upper_bound :int= 9_000_000 , start_index :int = 0, checkpoint_interval :int = 1_000_000 ) -> pd.DataFrame:
    """
    Fetches grades for Wikipedia pages from a dump object and returns the data as a DataFrame.

    Parameters:
    - dump (mwxml.Dump): A dump object representing the Wikipedia XML dump.
    - path (str): The path to the CSV file where the data and checkpoints will be saved.
    - upper_bound (int): An optional parameter specifying the maximum number of iterations to perform. Defaults to 9,000,000.
    - start_index (int): An optional parameter specifying the starting index. Defaults to 0.
    - checkpoint_interval (int): An optional parameter specifying the checkpoint interval for saving the data. Defaults to 1,000,000,.

    Returns:
    - df (pandas.DataFrame): A DataFrame containing the fetched data.
    """

    results = {}
    results["Title"] = []
    results["Grade"] = []
    results["RevID"] = []

    revids_storage = {}

    # Temporary storage for revision IDs
    chunk = []

    # Iterate over the dump with progress tracking
    for i, page in tqdm(enumerate(dump), total=upper_bound):
        
        #Skip until start index
        if i < start_index:
            continue
        
        #Skip redirects
        if page.redirect is not None:
            continue
        
        #Only retrieve articles 
        if page.namespace != 0:
            continue
        
        for revision in page:
            rev_id = str(revision.id)
            revids_storage[page.title] = rev_id
            chunk.append(rev_id) 
            break

        # Save the data to a CSV file at checkpoint intervals
        if i % checkpoint_interval == 0:
            df1 = pd.read_csv(f"{path}.csv")
            df2 = pd.DataFrame(results)
            df = pd.concat([df1, df2])
            df.to_csv(f"{path}.csv", index=False)
            
            #Free memory
            del df1
            del df2
            results = {"Title": [], "Grade": [], "RevID": []}
   
        if i > upper_bound:
            break
        
        # 50 is the limit for bulk requests on Wikipedia. 
        if len(chunk) == 50:
            rev_ids = "|".join(chunk)
            chunk = []
            for title, assessment in get_page_assessment(rev_ids):
                results["Title"].append(title)
                results["Grade"].append(assessment)
                try:
                    results["RevID"].append(revids_storage[title])
                except KeyError:
                    results["RevID"].append("Missing")
            revids_storage = {}
        else:
            continue

    # Convert the results dictionary to a DataFrame
    df1 = pd.read_csv(f"{path}.csv")
    df2 = pd.DataFrame(results)
    df = pd.concat([df1, df2])

    # Save the final DataFrame as a CSV file
    df.to_csv(f"{path}.csv", index=False)

    return df

def search_index(page_title, index_filename):
    """Searches for a page title in an index file and returns the start byte position and data length.

    Args:
        page_title (str): The title of the page to search for.
        index_filename (str): The filename of the index file.

    Returns:
        tuple: A tuple containing the start byte position and data length of the page.
            - start_byte (int): The starting byte position of the page in the index file.
            - data_length (int): The length of the page data.

    Source: https://alchemy.pub/wikipedia
    """

    byte_flag = False
    data_length = start_byte = 0
    index_file = open(index_filename, 'r', encoding="utf-8")
    csv_reader = csv.reader(index_file, delimiter=':')
    for line in csv_reader:
        if not byte_flag and page_title == line[2]:
            start_byte = int(line[0])
            byte_flag = True
        elif byte_flag and int(line[0]) != start_byte:
            data_length = int(line[0]) - start_byte
            break
    index_file.close()
    return start_byte, data_length



