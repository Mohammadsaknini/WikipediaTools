from mwparserfromhell.wikicode import Wikicode
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from ipaddress import ip_address 
from readcalc import readcalc
import networkx as nx
from enum import Enum
import numpy as np
import pytextrank
import statistics
import requests
import spacy
import json

headers = {"Accept-Encoding":"gzip"}

class Features(ABC):
    @abstractmethod
    def get_features(self) -> dict:
        """
        Abstract method to be implemented by subclasses.
        This method should return the features of the object as a dictionary.
        """
        pass

    def __str__(self) -> str:
        return json.dumps(self.get_features(), indent=4)

class TextStatisticsFeatures(Features):
    
    def __init__(self, code: Wikicode, nlp: spacy.language):
        """
        Initialize the TextFeatures.

        Args:
            code (Wikicode): The Wikicode object containing the text code.
        """
        raw_text = code.strip_code()
        doc = nlp(raw_text)

        sections = code.get_sections(flat=True)
        sentences = list(doc.sents)
        sections_length = self.sections_length(sections)
        self.features = {
            "Number of words": len(raw_text.split()),
            "Number of sentences": len(sentences),
            "Number of characters": len(raw_text),
            "Subsection count": len(code.get_sections(flat=True, levels=[3, 4, 5, 6])),
            "Section count": len(code.get_sections(flat=True, levels=[2])),
            "Average words number per paragraph": self.avg_words_per_paragraph(sections),
            "Average section length": sum(sections_length) / len(sections_length),
            "Word count of the longest sentence": self.word_count_of_longest_sentence(sentences),
            "Abstract size": len(sections[0].strip_code()),
            "Standard deviation of the section length": np.std(sections_length),
            "Size of largest section": max(sections_length),
            "Size of shortest section": min(sections_length)}

    def _get_paragraphs(self, subsection):
        """
        Split the subsection into paragraphs.

        Args:
            subsection (str): The subsection content.

        Returns:
            list: List of paragraphs.
        """
        return subsection.split("\n")

    def avg_words_per_paragraph(self, subsections):
        """
        Calculate the average number of words per paragraph in the given subsections.

        Args:
            subsections (list): List of subsections.

        Returns:
            float: Average number of words per paragraph.
        """
        length = []

        for subsection in subsections:
            for paragraph in self._get_paragraphs(subsection.strip_code()):
                length.append(len(paragraph.split()))

        return sum(length) / len(length)

    def sections_length(self, sections):
        """
        Calculate the length of each section in terms of the number of characters.

        Args:
            sections (list): List of sections.

        Returns:
            list: List of section lengths.
        """
        length = []
        for section in sections:
            length.append(len(section.strip_code()))
        return length

    def word_count_of_longest_sentence(self, sentences):
        """
        Calculate the word count of the longest sentence in the given list of sentences.

        Args:
            sentences (list): List of sentences.

        Returns:
            int: Word count of the longest sentence.
        """
        longest = 0
        for sentence in sentences:
            current = len(sentence.text.split())
            if current > longest:
                longest = current

        return longest

    def get_features(self):
        """
        Get the extracted text features.

        Returns:
            dict: Dictionary containing the text features.
        """
        return self.features

class StructureFeatures(Features):

    def __init__(self, code: Wikicode):
        """
        Initialize the StructureFeatures.

        Args:
            code (Wikicode): The Wikicode object containing the code.
        """
        self.code = code
        sections = code.get_sections(flat=True)
        citation_count = self.references_count()
        raw_text = code.strip_code()
        text_length = len(raw_text)
        total_links = len(code.filter_wikilinks())

        self.features = {
            "Average subsection number per section": self.avg_subsection_per_section(),
            "Citation number per section": self.citation_per_section(sections),
            "Citation count": citation_count,
            "Image number per section": self.images_per_section(sections),
            "Citation count per text length": citation_count / text_length,
            "Link count per section": self.links_per_section(sections),
            "Links per text length": total_links / text_length,
            "Number of external links": total_links,
        }

    def avg_subsection_per_section(self):
        """
        Calculate the average number of subsections per section.

        Returns:
            float: Average number of subsections per section.
        """
        sections_count = len(
            list(self.code.get_sections(flat=True, levels=[2])))
        sub_sections_count = len(
            list(self.code.get_sections(flat=True, levels=[3])))

        if sections_count == 0:
            return 0

        return (sub_sections_count / sections_count)

    def references_count(self):
        """
        Count the number of references in the code.

        Returns:
            int: Number of references.
        """
        count = 0
        for tag in self.code.filter_tags():
            if tag.tag == "ref" or "cite" in tag.tag and tag.contents != "":
                count += 1
        return count

    def citation_per_section(self, sections):
        """
        Calculate the average number of citations per section.

        Args:
            sections (list): List of sections.

        Returns:
            float: Average number of citations per section.
        """
        count = 0
        for section in sections:
            for tag in section.filter_tags():
                if "ref" in tag.tag or "cite" in tag.tag:
                    count += 1

        return count / len(list(sections))

    def images_per_section(self, sections):
        """
        Calculate the average number of images per section.

        Args:
            sections (list): List of sections.

        Returns:
            float: Average number of images per section.
        """
        count = 0
        for section in sections:
            for link in section.filter_wikilinks():
                title = link.title.lower()
                if "File:" in title or ".png" in title or ".jpg" in title or ".gif" in title or ".jpeg" in title:
                    count += 1

        return count / len(list(sections))

    def links_per_section(self, sections):
        """
        Calculate the average number of links per section.

        Args:
            sections (list): List of sections.

        Returns:
            float: Average number of links per section.
        """
        links = 0
        for section in sections:
            links += len(list(section.filter_wikilinks()))
        return links / len(list(sections))

    def get_features(self):
        """
        Get the structure features.

        Returns:
            dict: Dictionary containing the structure features.
        """
        return self.features

class ReadabilityFeatures(Features):

    def __init__(self, code: Wikicode):
        """
        Initialize the ReadabilityFeatures.

        Args:
            code (Wikicode): The Wikicode object containing the code.
        """
        self.code = code
        raw_text = code.strip_code()
        calc = readcalc.ReadCalc(raw_text, preprocesshtml="justext")
        self.features = {
            "Automated readability index (ARI)": calc.get_ari_index(),
            "Coleman-Liau Index": calc.get_coleman_liau_index(),
            "Flesch reading ease": calc.get_flesch_reading_ease(),
            "Flesch-Kincaid Grade Level": calc.get_flesch_kincaid_grade_level(),
            "Gunning Fog Index": calc.get_gunning_fog_index(),
            "LIX": calc.get_lix_index(),
            "SMOG score": calc.get_smog_index(),
        }

    def get_features(self):
        """
        Get the readability features.

        Returns:
            dict: Dictionary containing the readability features.
        """
        return self.features

class WritingStyleFeatures(Features):

    def __init__(self, code: Wikicode, nlp: spacy.language):
        """
        Initialize the WritingStyleFeatures.

        Args:
            code (Wikicode): The Wikicode object containing the code.
            nlp (spacy.language): The spacy language model.

        """
        raw_text = code.strip_code()
        number_of_words = len(raw_text.split())
        doc = nlp(raw_text)
        sentences = list(doc.sents)
        total_pharses = len(doc._.phrases)
        doc_features = self.doc_features(doc)
        sentences_features = self.sentence_features(sentences)

        self.features = {
            "Number of passive sentences": doc_features["passive_count"],
            "Number of questions": sentences_features["question_count"],
            "Long phrase rate": self.count_long_phrases(doc, threshold=6) / total_pharses,
            "Short phrase rate": self.count_short_phrases(doc, threshold=3) / total_pharses,
            "Auxiliary verb number": doc_features["auxiliary_verb_count"],
            "Conjunction rate": doc_features["conjunction_count"] / number_of_words,
            "Sentence number with pronoun as beginning": sentences_features["pronoun_count (start)"],
            "Sentence number with article as beginning": sentences_features["article_count"],
            "Sentence number with conjunction as beginning": sentences_features["conjunction_count"],
            "Sentence number with subordinating conjunction as beginning": sentences_features["subordinating_conjunction_count"],
            "Sentence number with interrogative pronoun as beginning": sentences_features["interrogative_pronoun_count"],
            "Sentence number with preposition as beginning": sentences_features["preposition_count"],
            "Nominalization rate": doc_features["nominalization_rate"],
            "Preposition rate": doc_features["preposition_count"] / number_of_words,
            "'To be' verb rate": doc_features["to_be_verb_rate"],
            "Number of pronuns": doc_features["pronoun_count"],
        }
    
    def count_long_phrases(self, doc, threshold=10):
        """
        Count the number of long phrases in the given text based on the specified threshold.

        Parameters:
        doc (spacy.Doc): The input text processed by SpaCy.
        threshold (int, optional): The minimum length of a phrase to be considered as "long".
            Default is 10.

        Returns:
        int: The total number of long phrases in the text.
        """
        # Process the text using SpaCy

        long_phrase_count = 0
        current_long_phrase_length = 0

        for token in doc:
            if not token.is_punct:
                current_long_phrase_length += 1
            else:
                if current_long_phrase_length >= threshold:
                    long_phrase_count += 1
                
                current_long_phrase_length = 0

        # Check if there is a remaining long phrase at the end of the text
        if current_long_phrase_length >= threshold:
            long_phrase_count += 1

        return long_phrase_count

    def count_short_phrases(self, doc, threshold=5):
        """
        Count the number of short phrases in the given text based on the specified threshold.

        Parameters:
        doc (spacy.Doc): The input text processed by SpaCy.
        threshold (int, optional): The maximum length of a phrase to be considered as "short".
            Default is 2.

        Returns:
            int: The total number of short phrases in the text.
        """

        short_phrase_count = 0
        current_short_phrase_length = 0

        for token in doc:
            if not token.is_punct:
                current_short_phrase_length += 1
            else:
                if current_short_phrase_length <= threshold:
                    short_phrase_count += 1
                
                current_short_phrase_length = 0

        # Check if there is a remaining short phrase at the end of the text
        if current_short_phrase_length <= threshold:
            short_phrase_count += 1

        return short_phrase_count
    
    def doc_features(self, doc):
        """
        Count the number of auxiliary verbs, passive sentences, and conjunctions in the given text.
        
        Parameters:
        doc (spacy.Doc): The input text processed by SpaCy.

        Returns:
            dict: Dictionary containing the counts of auxiliary verbs, passive sentences, and conjunctions.
        """
        to_be_verbs = ("am", "is", "are", "was", "were", "been", "being")
        verb_count = 0
        to_be_verbs_count = 0

        auxiliary_verb_count = 0
        passive_count = 0
        conjunction_count = 0
        noun_count = 0
        nominalization_count = 0
        pronouns_count = 0
        preposition_count = 0

        for token in doc:

            if token.pos_ == "CCONJ" or token.pos_ == "SCONJ":
                conjunction_count += 1

            if token.pos_ == "AUX":
                auxiliary_verb_count += 1

            elif token.dep_ == "nsubjpass":
                passive_count += 1

            elif token.pos_ == "PRON":
                pronouns_count += 1

            elif token.pos_ == "ADP":
                preposition_count += 1

            elif token.pos_ == "VERB":
                verb_count += 1
                if token.text.lower() in to_be_verbs:
                    to_be_verbs_count += 1

            elif token.pos_ == "NOUN":
                noun_count += 1
                if token.lemma_ != token.text.lower():
                    nominalization_count += 1


        if noun_count == 0 :
            nominalization_rate = 0
        else:
            nominalization_rate = nominalization_count / noun_count
        
        if verb_count == 0 :
            to_be_verb_rate = 0
        else:    
            to_be_verb_rate = to_be_verbs_count / verb_count

        
        
        results = {"auxiliary_verb_count": auxiliary_verb_count,
                   "passive_count": passive_count,
                   "conjunction_count": conjunction_count,
                   "nominalization_rate": nominalization_rate,
                   "to_be_verb_rate": to_be_verb_rate,
                   "pronoun_count": pronouns_count,
                   "preposition_count": preposition_count}
        return results

    def sentence_features(self, sentences):
        pronoun_count = 0
        article_count = 0
        conjunction_count = 0
        subordinating_conjunction_count = 0
        interrogative_pronoun_count = 0
        preposition_count = 0
        question_count = 0

        for sent in sentences:
            first_token = sent[0]

            if "?" in sent.text:
                question_count += 1

            if first_token.text.lower() in ["who", "whom", "whose", "which", "what"]:
                interrogative_pronoun_count += 1

            if first_token.pos_ == "CCONJ":
                conjunction_count += 1

            elif first_token.pos_ == "PRON":
                pronoun_count += 1

            elif first_token.pos_ == "DET":
                article_count += 1

            elif first_token.pos_ == "SCONJ":
                subordinating_conjunction_count += 1
                conjunction_count += 1

            elif first_token.pos_ == "ADP":
                preposition_count += 1

        results = {
                "pronoun_count (start)": pronoun_count,
                "article_count": article_count,
                "conjunction_count": conjunction_count,
                "subordinating_conjunction_count": subordinating_conjunction_count,
                "interrogative_pronoun_count": interrogative_pronoun_count,
                "preposition_count": preposition_count,
                "question_count": question_count
                   }
        return results

    def get_features(self):
        """
        Get the writing style features.

        Returns:
            dict: Dictionary containing the writing style features.
        """
        return self.features

class EditHistoryFeatures(Features, ABC):
    def __init__(self, revid):
        self.revid = revid

    def mean_time_between_reviews(self, timestamps, days = 30):
        
        if not timestamps:
            return -1
        
        current_date = datetime.utcnow()
        past_30_days = current_date - timedelta(days=days)

        # Filter timestamps to include only those from the past 30 days
        timestamps = [ts for ts in timestamps if ts >= past_30_days]

        if len(timestamps) <= 1:
            return -1

        time_diffs = [(timestamps[i - 1] - timestamps[i]).total_seconds() for i in range(1, len(timestamps))]
        mean_time = sum(time_diffs) / len(time_diffs)

        return mean_time / 86400 # Convert to days

    def calculate_age(self, creation_timestamp, current_timestamp):
        return 

    def average_edits_per_user_and_anonymous_count(self, unique_users, total_edits, anonymous_ips):

        if total_edits == 0:
            return 0
        avg_edits = total_edits / unique_users

        return avg_edits, anonymous_ips
    
    def _validIPAddress(self, ip):
        try:
            ip_address(ip)
            return True
        except ValueError:
            return False
        
    def occasional_users_rate(self, edits_per_user):

        total_reviews = 0
        reviews_less_than_4 = 0

        for _, edits in edits_per_user.items():
            total_reviews += edits
            if edits < 4:
                reviews_less_than_4 += edits

        if total_reviews == 0:
            return 0

        percentage = (reviews_less_than_4 / total_reviews)

        return percentage

    def get_review_count_past_3_months(self, timestamps):

        three_months_ago = datetime.utcnow() - timedelta(days=90)
        count = 0

        for timestamp in timestamps:
            if timestamp >= three_months_ago:
                count += 1

        return count
    
    def percentage_reviews_top_5(self, edits_per_user):

        total_edits = sum(list(edits_per_user.values()))
        sorted_edits_counts = {k: v for k, v in sorted(edits_per_user.items(), key=lambda item: item[1], reverse=True)}
        top_5_reviewers = int(len(sorted_edits_counts) * 0.05)
        total_reviews_top_5 = sum(list(sorted_edits_counts.values())[:top_5_reviewers])
        percentage_top_5 = (total_reviews_top_5 / total_edits)

        return percentage_top_5
    
    def get_discussion_count(self):
        params = {
            'action': 'query',
            'format': 'json',
            'titles': f'Talk:{self.page_title}',
            'prop': 'revisions',
            'rvprop': 'timestamp',
            'rvlimit': 'max'
        }

        discussion_count = 0

        while True:
            response = requests.get(self.base_url, params=params, headers=headers)
            data = response.json()

            page_id = next(iter(data['query']['pages']))  # Get the page ID
            revisions = data['query']['pages'][page_id]['revisions']

            discussion_count += len(revisions)

            if 'continue' in data:
                params['rvcontinue'] = data['continue']['rvcontinue']
            else:
                break

        return discussion_count

    def get_article_age(self, timestamps):
        timestamps = sorted(timestamps)
        current_timstamp = timestamps[0]
        creation_timestamp = timestamps[-1]
        return (creation_timestamp - current_timstamp).days - 1
    
    @abstractmethod
    def get_revision_timestamps(self):
        pass
    @abstractmethod
    def get_users_and_edit_count(self):
        pass

    @abstractmethod
    def modified_size_rate(self):
        pass

class EditHistoryFeaturesDump(EditHistoryFeatures):
    def __init__(path) -> None:
        
        pass

    def get_revision_timestamps(self):
        pass
    
    def get_article_age(self):
        pass
    
    def get_users_and_edit_count(self):
        pass
    
    def get_anonymous_ips(self):
        pass
    
    def get_edits_per_user(self):
        pass
    
    def modified_size_rate(self):
        pass

class EditHistoryFeaturesAPI(EditHistoryFeatures):

    def __init__(self, page_title, revid):
        """
        Initialize the edit history features.

        Parameters:
            page_title (str): The title of the Wikipedia page.
        """
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.page_title = page_title
        if isinstance(revid, str):
            revid = int(revid)
        self.revid = revid
        total_users, total_reviews , ips_count, edits_per_user = self.get_users_and_edit_count()
        avg_edits_per_user = total_reviews / total_users
        timestamps = self.get_revision_timestamps()
        article_age = self.get_article_age(timestamps)

        self.features = {
            "Age": article_age,
            "Mean time between two reviews (30 days)": self.mean_time_between_reviews(timestamps, 30),
            "Average edits per user": avg_edits_per_user,
            "Discussion count": self.get_discussion_count(),
            "IP number": ips_count,
            "Review count": total_reviews,
            "User count": total_users - ips_count,
            "Modified size rate": self.modified_size_rate(),
            "Occasional user review rate": self.occasional_users_rate(edits_per_user),
            "Average review rate last three months": self.get_review_count_past_3_months(timestamps) / len(timestamps),
            "Most active user review rate": self.percentage_reviews_top_5(edits_per_user),
            "Standard deviation of edit number per user": np.std(list(edits_per_user.values())),
            "Reviews number per day": total_reviews / article_age,
        }

    def get_revision_timestamps(self):
        params = {
            'action': 'query',
            'format': 'json',
            'titles': self.page_title,
            'prop': 'revisions',
            'rvprop': 'timestamp|ids',
            'rvlimit': 'max'
        }
        timestamps = []
        while True:

            response = requests.get(self.base_url, params=params, headers=headers)
            data = response.json()

            page_id = next(iter(data['query']['pages']))
            revisions = data['query']['pages'][page_id]['revisions']

            for rev in revisions:
                rev_id = rev['revid']
                if self.revid >= rev_id:
                    timestamps.append(datetime.strptime(rev['timestamp'], "%Y-%m-%dT%H:%M:%SZ"))
            
            if 'continue' in data:
                params['rvcontinue'] = data['continue']['rvcontinue']
            else:
                break

        return timestamps
 
    def get_users_and_edit_count(self):
        """
        Get the number of users and edits for the page.

        Returns:
            tuple: A tuple containing the number of users, the number of edits, the number of anonymous IPs
                   and a dictionary containing the number of edits for each user.
        """
        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'revisions',
            'titles': self.page_title,
            'rvprop': 'user|ids',
            'rvlimit': 'max'
        }

        users = set()
        anonymous_ips = set()
        total_edits = 0
        edits_by_user = {}

        while True:
            response = requests.get(self.base_url, params=params, headers=headers)
            data = response.json()

            page_id = next(iter(data['query']['pages']))  # Get the page ID
            revisions = data['query']['pages'][page_id]['revisions']

            if not revisions:
                break

            total_edits += len(revisions)
            
            for rev in revisions:
                
                if rev['revid'] > self.revid:
                    continue

                if 'user' in rev:
                    user = rev['user']
                    users.add(user)
                    edits_by_user[user] = edits_by_user.get(user, 0) + 1

                if 'user' in rev and self._validIPAddress(rev['user']):
                    anonymous_ips.add(rev['user'])

            if 'continue' in data:
                params['rvcontinue'] = data['continue']['rvcontinue']
            else:
                break
        
        return len(users), total_edits, len(anonymous_ips), edits_by_user

    def modified_size_rate(self):

        params = {
            'action': 'query',
            'format': 'json',
            'prop': 'revisions',
            'revids': self.revid,
            'rvprop': 'timestamp|size',
        }

        response = requests.get(self.base_url, params=params, headers=headers)
        data = response.json()

        page_id = next(iter(data['query']['pages']))  # Get the page ID
        revisions = data['query']['pages'][page_id]['revisions']

        if len(revisions) >= 1:
            current_revision = revisions[0]['size']

            # Get the timestamp of the current revision
            current_timestamp = datetime.strptime(revisions[0]['timestamp'], "%Y-%m-%dT%H:%M:%SZ")

            # Get the timestamp of the revision roughly three months ago
            three_months_ago = current_timestamp - timedelta(days=90)
            three_months_ago_str = three_months_ago.strftime('%Y-%m-%dT%H:%M:%SZ')

            # Fetch the revision ID of the version roughly three months ago
            params['rvstart'] = three_months_ago_str
            params.pop('revids')
            params['titles'] = self.page_title

            response = requests.get(self.base_url, params=params, headers=headers)
            data = response.json()
            revisions = data['query']['pages'][page_id]['revisions']

            if len(revisions) >= 1:
                previous_revision = revisions[0]['size']

                return current_revision - previous_revision / current_revision

        return -1

    def get_features(self):
        """
        Get the edit history features.

        Returns:
            dict: Dictionary containing the edit history features.
        """
        return self.features

class NetworkFeatures(Features):
    
    def __init__(self, page_title, code: Wikicode):
        self.base_url = 'https://en.wikipedia.org/w/api.php'
        self.page_title = page_title
        in_degree = self.get_in_degree() 
        out_degree = self.get_out_degree()
        graph = self.create_graph(in_degree, out_degree)
        self.features = {
        "In Degree":len(in_degree) if isinstance(in_degree, list) else in_degree,
        "Out Degree":len(out_degree),
        "Page Rank":nx.pagerank(graph).get(self.page_title),
        "Reciprocity":nx.reciprocity(graph),
        "Links":len(code.filter_wikilinks()),
        "Translation count":self.get_translation_count(),
        "Associativity of the in-degree and in-degree":self.calculate_assortativity(graph, "in", "in"),
        "Associativity of the in-degree and out-degree":self.calculate_assortativity(graph, "in", "out"),
        "Associativity of the out-degree and in-degree":self.calculate_assortativity(graph, "out", "in"),
        "Associativity of the out-degree and out-degree":self.calculate_assortativity(graph, "out","out"),
        }

    def calculate_assortativity(self, graph: nx.DiGraph, node_degree, neighbor_degree):
        if node_degree == "in":
            node = dict(graph.in_degree).get(self.page_title)
        else:
            node = dict(graph.out_degree).get(self.page_title)
        neigbors_avg_degree = statistics.mean(list(nx.average_neighbor_degree(graph, neighbor_degree,).values()))
        
        if neigbors_avg_degree == 0:
            return 0
        return node/neigbors_avg_degree

    def get_in_degree(self):

        params = {
            'action': 'query',
            'titles': self.page_title,
            'prop': 'linkshere',
            'lhlimit': 'max',
            'format': 'json',
            'lhnamespace': '0'
        }
        in_degree = []
        try:
            while True:
                response = requests.get(self.base_url, params=params, headers=headers)
                data = response.json()

                page_id = next(iter(data['query']['pages'].keys()))
                in_degree += [link['title'] for link in data['query']['pages'][page_id]['linkshere']]

                if 'continue' in data:
                    params['lhcontinue'] = data['continue']['lhcontinue']
                else:
                    break
        except:
            return 0

        return in_degree
    
    def get_out_degree(self):

        params = {
            'action': 'query',
            'titles': self.page_title,
            'prop': 'links',
            'pllimit': 'max',
            'format': 'json',
            'plnamespace': '0'
        }
        out_degree = []

        while True:
            response = requests.get(self.base_url, params=params, headers=headers)
            data = response.json()

            page_id = next(iter(data['query']['pages'].keys()))
            out_degree += [link['title'] for link in data['query']['pages'][page_id]['links']]

            if 'continue' in data:
                params['plcontinue'] = data['continue']['plcontinue']
            else:
                break

        return out_degree
    
    def create_graph(self, in_degree, out_degree):
        # Create an empty directed graph
        graph = nx.DiGraph()
        graph.add_node(self.page_title)

        for degree in out_degree:
            graph.add_edge(self.page_title, degree, directed=True)

        if isinstance(in_degree, list):
            for degree in in_degree:
                graph.add_edge(degree, self.page_title, directed=True)

        return graph
    
    def get_translation_count(self):

        params = {
            'action': 'query',
            'titles': self.page_title,
            'prop': 'langlinks',
            'lllimit': 'max',
            'format': 'json'
        }

        response = requests.get(self.base_url, params=params)
        response_data = response.json()

        page_data = next(iter(response_data['query']['pages'].values()))

        if 'langlinks' in page_data:
            return len(page_data['langlinks'])
        else:
            return 0  # No translations found

    def get_features(self):
        return self.features

class FeatureSets(Enum):
    
    TEXT_STATISTICS = [
    "Number of words",
    "Number of sentences",
    "Number of characters",
    "Subsection count",
    "Section count",
    "Average words number per paragraph",
    "Average section length",
    "Word count of the longest sentence",
    "Abstract size",
    "Standard deviation of the section length",
    "Size of largest section",
    "Size of shortest section"]

    STRUCTURE = [
    "Average subsection number per section",
    "Citation number per section",
    "Citation count",
    "Image number per section",
    "Citation count per text length",
    "Link count per section",
    "Links per text length",
    "Number of external links"]

    READABILITY_SCORES = [
    "Automated readability index (ARI)",
    "Coleman-Liau Index",
    "Flesch reading ease",
    "Flesch-Kincaid Grade Level",
    "Gunning Fog Index",
    "LIX",
    "SMOG score"]
    
    WRITING_STYLE = [
        "Number of passive sentences",
        "Number of questions",
        "Long phrase rate",
        "Short phrase rate",
        "Auxiliary verb number",
        "Conjunction rate",
        "Sentence number with pronoun as beginning",
        "Sentence number with article as beginning",
        "Sentence number with conjunction as beginning",
        "Sentence number with subordinating conjunction as beginning",
        "Sentence number with interrogative pronoun as beginning",
        "Sentence number with preposition as beginning",
        "Nominalization rate",
        "Preposition rate",
        "'To be' verb rate",
        "Number of pronuns"
    ]
    
    EDIT_HISTORY = [
        "Age",
        "Mean time between two reviews (30 days)",
        "Average edits per user",
        "Discussion count",
        "IP number",
        "Review count",
        "User count",
        "Modified size rate",
        "Occasional user review rate",
        "Average review rate last three months",
        "Most active user review rate",
        "Standard deviation of edit number per user",
        "Reviews number per day"
    ]
    
    NETWORK = [
        "In Degree",
        "Out Degree",
        "Page Rank",
        "Reciprocity",
        "Links",
        "Translation count",
        "Associativity of the in-degree and in-degree",
        "Associativity of the in-degree and out-degree",
        "Associativity of the out-degree and in-degree",
        "Associativity of the out-degree and out-degree",
        ]
    
    _EMBEDDINGS = [f"Embedding_{i}" for i in range(1,301)]
    
    ALL = TEXT_STATISTICS + STRUCTURE + READABILITY_SCORES + WRITING_STYLE + EDIT_HISTORY + NETWORK
    _CUSTOM = ALL + _EMBEDDINGS

class FeaturesExtractor:

    def __init__(self, feature_sets: [FeatureSets] = FeatureSets.ALL) -> None:
        """
        Initializes the features extractor.
        
        Args:
            feature_sets ([FeatureSets], optional): The feature sets to extract. Defaults to FeatureSets.ALL.
        """
        self.feature_sets = feature_sets
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textrank")
        spacy.prefer_gpu()

    def extract_feature_sets(self, page_title: str, revid: str, code: Wikicode) -> dict:
        """
        Extracts various feature sets from the provided code.

        Args:
            page_title (str): The title of the Wikipedia page.
            revid (str): The revision ID of the Wikipedia page.
            code (mwparserfromhell.Wikicode): The parsed Wikipedia code to extract features from.

        Returns:
            dict: A dictionary containing the extracted feature sets.
        """
        features = {}
        if isinstance(self.feature_sets, FeatureSets):
            self.feature_sets = [self.feature_sets]
        try:
            if FeatureSets.ALL in self.feature_sets:
                features.update(TextStatisticsFeatures(code, self.nlp).features)
                features.update(StructureFeatures(code).features)
                features.update(ReadabilityFeatures(code).features)
                features.update(WritingStyleFeatures(code, self.nlp).features)
                features.update(EditHistoryFeaturesAPI(page_title, revid).features)
                features.update(NetworkFeatures(page_title, code).features)
                return features

            if FeatureSets.TEXT_STATISTICS in self.feature_sets:
                features.update(TextStatisticsFeatures(code, self.nlp).features)

            if FeatureSets.STRUCTURE in self.feature_sets:
                features.update(StructureFeatures(code).features)

            if FeatureSets.READABILITY_SCORES in self.feature_sets:
                features.update(ReadabilityFeatures(code).features)

            if FeatureSets.WRITING_STYLE in self.feature_sets:
                features.update(WritingStyleFeatures(code, self.nlp).features)

            if FeatureSets.EDIT_HISTORY in self.feature_sets:
                features.update(EditHistoryFeaturesAPI(page_title, revid).features)

            if FeatureSets.NETWORK in self.feature_sets:
                features.update(NetworkFeatures(page_title, code).features)
        except ValueError:
            raise ValueError("Features could not be extracted. Please try a different page.")

        return features
