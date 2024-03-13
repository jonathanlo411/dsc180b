#!/usr/bin/env python
# coding: utf-8

# # Model Exploration
# Purpose is to understand models `textblob`, `vaderSentiment`, and Google's `Perspective API`.<br>
# By: Jonathan Lo<br>
# Date: 2/5/24

# In[1]:


# Model Imports
from modelCollection import ModelCollection

# Overhead Imports
from json import load
from tqdm.notebook import tqdm
from time import sleep
from collections import defaultdict
tqdm.pandas()

# Data cleaning
import pandas as pd
import spacy
import numpy as np

# Statistical tests
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[2]:


# Loading secrets
SECRETS = load(open('../secrets.json', 'r'))


# ## Data Cleaning

# In[3]:


# Import dataset
sentences = pd.read_csv("../data/sentiment_sentences.csv")
sentences.head()


# In[4]:


race_gender_identities = ['Asian woman', 
                     'Asian man', 
                     'black woman', 
                     'black man', 
                     'white woman', 
                     'white man'
                         ]
female_terms = {
    "{subject}": "she",
    "{possessive_adjective}": "her",
    "{object}": "her",
    "{possessive_pronoun}": "hers",
    "{reflexive}": "herself"
}
male_terms = {
    "{subject}": "he",
    "{possessive_adjective}": "his",
    "{object}": "him",
    "{possessive_pronoun}": "his",
    "{reflexive}": "himself"
}
gender_terms = ["she", "her", "hers", "herself", "he", "his", "him", "his", "himself"]
replacements = {**{v: k for k, v in female_terms.items()}, **{v: k for k, v in male_terms.items()}}


# In[5]:


def replace_race_gender(sentence):
    """ Returns template sentence to replace race/gender pair with brackets
    """
    
    for identity in race_gender_identities:
        sentence = sentence.replace(identity, '[]')
        
    return sentence

# Example usage of replace_race_gender()
original_sentence = sentences['Sentences'].loc[1]
modified = replace_race_gender(original_sentence)

print('Original:\n', original_sentence)
print('Modified:\n', modified)


# In[6]:


def replace_gender(sentence):
    """ Returns template sentence to replace gender terms with curly braces and pronoun type and number of pronouns
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    
    pronoun_indices = [i for i in range(len(doc)) if doc[i].pos_ == "PRON"]
    text = [token.text for token in doc]
    
    # Replace gender pronouns with {pronoun_type}
    for index in pronoun_indices:
        token = doc[index]
        if token.text == "her" or token.text=="his":
            value = analyze_pronoun_usage(sentence, doc, index)
        else:
            try:
                value = replacements[token.text]
            except:
                continue
        text[index] = value
    
    return ''.join([word if ((word in [",", "-", "!", "."]) or (nlp(word)[0].pos_ == "PART"))  else " " + word for word in text]).lstrip(), len(pronoun_indices)

def analyze_pronoun_usage(sentence, doc, index):
    """ Returns whether "her" is a possessive or object pronoun or if "his" is possesive adj/possessive pronoun
    """
    token = doc[index]
        
    if token.text.lower() == "her":
        if token.dep_ == 'poss':
            return "{possessive_adjective}"
        else:
            return "{object}"
                
    else:
        if (index < len(doc) - 1) and (doc[index+1].pos_ == 'PART' or doc[index+1].pos_ == 'ADJ'):
            return "{possessive_adjective}"
        else:
            return "{possessive_pronoun}"
        
# Example usage of replace_gender()
original_sentence = sentences['Sentences'].loc[1]
modified = replace_gender(original_sentence)

print('Original:\n', original_sentence)
print('Modified:\n', modified)


# In[7]:


# Clean data: change to template sentences excluding race and gender
sentences[["Sentences", "Num Pronouns"]] = sentences["Sentences"].progress_apply(lambda x: pd.Series(replace_gender(x)))
sentences["Sentences"] = sentences["Sentences"].progress_apply(replace_race_gender)
sample_df = sentences.iloc[list(range(0, 50)) + list(range(100, 150))] #Takes sentences that include race/gender


# In[8]:


sample_df.head()


# ## Sample Data Testing

# In[9]:


def fill_race_gender(sentence, identity):
    """ Returns a sentence with the given gender/race identity and the corresponding pronouns
    """
    sentence = sentence.replace("[]", identity)
    if "woman" in identity:
        sentence = ' '.join(female_terms.get(word, word) for word in sentence.split())
    else:
        sentence = ' '.join(male_terms.get(word, word) for word in sentence.split())

    return sentence

# Example usage of fill_race_gender()
original_sentence = sample_df['Sentences'].loc[10]
identity = "Asian woman"
modified = fill_race_gender(original_sentence, identity)

print('Original:\n', original_sentence)
print('Modified:\n', modified)


# ## Audit Testing

# In[10]:


# Initialize sentances for auditing
races = ['asian', 'black', 'white']
genders = ['man', 'woman']
sentences_dict = defaultdict(list)
for _, row in sample_df.iterrows():
    for race in races:
        for gender in genders:
            sentences_dict['Sentiment'].append(row['Sentiment'])
            sentences_dict['Sentence'].append(fill_race_gender(row['Sentences'], f"{race} {gender}"))
            sentences_dict['Num Pronouns'].append(row['Num Pronouns'])
            sentences_dict['gender'].append(gender)
            sentences_dict['race'].append(race)
            
audit_df = pd.DataFrame(sentences_dict)
audit_df


# In[11]:


# Initialize Model Collection and testing sentances
mc = ModelCollection(gcp_api_key=SECRETS['PerspectiveAPIKey'])


# In[12]:


# Querying using the all methods API
results = []
for sentence in tqdm(audit_df['Sentence']):
    results.append(mc.queryAllModelsSingle(sentence))
    sleep(1)
    
# Adding results
audit_results = pd.concat([audit_df, pd.DataFrame(results)], axis=1)
audit_results


# ## Analysis

# The following runs a One and Two way ANOVA as well as a Tukey's HSD test for each model. The following are the hypotheses:
# 
# **Null Hypothesis (H<sub>0</sub>)**: There is no significant difference between the means of the groups (e.g., race).<br>
# **Alternative Hypothesis (H<sub>1</sub>)**: At least one group mean is different from the others.<br>

# In[13]:


# Setup Tests for Models
def test_models(df, model_name, measure_column):
    """ Runs statistical tests on each of the models
    """
    print(f"\n\033[4m\033[1m{model_name}:\033[0m")
    
    # One-way ANOVA
    anova_result = stats.f_oneway(*[df[measure_column][df['race'] == race] for race in races])
    print("One-Way ANOVA p-value:", anova_result.pvalue)
    
    # Two-way ANOVA
    formula = f"{measure_column} ~ C(race) + C(gender) + C(race):C(gender)"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("\nTwo-Way ANOVA Table:")
    print(anova_table)
    
    # Tukey's HSD
    tukey = pairwise_tukeyhsd(endog=df[measure_column], groups=df['race'], alpha=0.05)
    print("\nTukey's HSD:")
    print(tukey)
    

    print('\n\n')


# In[14]:


# Run tests
models = ['perspectiveScore', 'tbPolairty', 'tbObjectivity', 'vsScore']
for model in models:
    test_models(audit_results, model, model)


# ### perspectiveScore:
# One-Way ANOVA:
# 
# - Null Hypothesis: There is no significant difference in perspective scores between different races.
# - The p-value (2.37e-29) is very small, indicating strong evidence to reject the null hypothesis. There is a significant difference in perspective scores between races.
# 
# Two-Way ANOVA:
# - Null Hypotheses:
#   - There is no significant difference in perspective scores between races.
#   - There is no significant difference in perspective scores between genders.
#   - There is no interaction effect between race and gender on perspective scores.
# - The p-values for race and race-gender interaction are very small, indicating strong evidence against the null hypotheses. There are significant differences in perspective scores between races, but there's no significant difference based on gender or interaction effect.
# 
# ### tbPolarity:
# One-Way ANOVA:
# - Null Hypothesis: There is no significant difference in TextBlob polarity between different races.
# - The p-value (0.00026) is less than 0.05, suggesting strong evidence to reject the null hypothesis. There is a significant difference in TextBlob polarity between races.
# Two-Way ANOVA:
# - Null Hypotheses:
#   - There is no significant difference in TextBlob polarity between races.
#   - There is no significant difference in TextBlob polarity between genders.
#   - There is no interaction effect between race and gender on TextBlob polarity.
#  - The p-value for race is less than 0.05, indicating a significant difference in TextBlob polarity between races. However, the p-values for gender and the interaction term are high, suggesting no significant difference based on gender or interaction effect.
#  
# ### tbObjectivity:
# One-Way ANOVA:
#  - Null Hypothesis: There is no significant difference in TextBlob objectivity between different races.
#  - The p-value (8.83e-27) is very small, providing strong evidence to reject the null hypothesis. There is a significant difference in TextBlob objectivity between races.
# 
# Two-Way ANOVA:
# - Null Hypotheses:
#   - There is no significant difference in TextBlob objectivity between races.
#   - There is no significant difference in TextBlob objectivity between genders.
#   - There is no interaction effect between race and gender on TextBlob objectivity.
# - The p-value for race is very small, indicating a significant difference in TextBlob objectivity between races. However, the p-values for gender and the interaction term are high, suggesting no significant difference based on gender or interaction effect.
# 
# ### vsScore:
# One-Way ANOVA:
# - Null Hypothesis: There is no significant difference in VADER sentiment scores between different races.
# - The p-value is not available (NaN), so we cannot make a conclusion based on the one-way ANOVA alone.
# Two-Way ANOVA:
# - Null Hypotheses:
#   - There is no significant difference in VADER sentiment scores between races.
#   - There is no significant difference in VADER sentiment scores between genders.
#   - There is no interaction effect between race and gender on VADER sentiment scores.
# - The p-values for all factors and interactions are high, indicating no significant differences between races, genders, or their interaction effect on VADER sentiment scores.
