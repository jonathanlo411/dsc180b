#!/usr/bin/env python
# coding: utf-8

# # Model Exploration
# Purpose is to understand models `textblob`, `vaderSentiment`, and Google's `Perspective API`.<br>
# By: Elsie Wang<br>
# Date: 2/7/24

# In[31]:


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
import matplotlib.pyplot as plt

# Statistical analysis
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Loading secrets
SECRETS = load(open('../sample.secrets.json', 'r'))


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


# In[8]:


sentences.head()


# In[9]:


# Plot Num Pronoun Distribution
plt.hist(sentences['Num Pronouns'], range=(0,6), bins=6)

plt.xlabel('Number of Pronouns')
plt.ylabel('Frequency')
plt.title("Sentence Pronoun Distribution")


# In[10]:


# Group by sentiment label
sentences['Sentiment'].value_counts()


# ## Sample Data Testing

# In[11]:


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
original_sentence = sentences['Sentences'].loc[30]
identity = "Asian woman"
modified = fill_race_gender(original_sentence, identity)

print('Original:\n', original_sentence)
print('Modified:\n', modified)


# ## Audit Testing

# In[12]:


# Initialize sentences for auditing
races = ['asian', 'black', 'white']
genders = ['man', 'woman']
sentences_dict = defaultdict(list)
for _, row in sentences.iterrows():
    for race in races:
        for gender in genders:
            sentences_dict['Sentiment'].append(row['Sentiment'])
            sentences_dict['Sentence'].append(fill_race_gender(row['Sentences'], f"{race} {gender}"))
            sentences_dict['Num Pronouns'].append(row['Num Pronouns'])
            sentences_dict['Gender'].append(gender)
            sentences_dict['Race'].append(race)
            
audit_df = pd.DataFrame(sentences_dict)
audit_df


# In[13]:


# Initialize Model Collection and testing sentances
mc = ModelCollection(gcp_api_key=SECRETS['PerspectiveAPIKey'])


# In[14]:


# Querying using the all methods API
results = []
for sentence in tqdm(audit_df['Sentence']):
    results.append(mc.queryAllModelsSingle(sentence))
    sleep(1)
    
# Adding results
audit_results = pd.concat([audit_df, pd.DataFrame(results)], axis=1)
audit_results


# In[58]:


# Make perspectiveScore range from -1 to 1
audit_results['perspectiveScore_normalized'] = 1 - 2 * audit_results['perspectiveScore']


# In[69]:


# Plot Score Distributions
plt.hist(audit_results['tbPolairty'], range=(-1,1), alpha=0.6, label='textblob')
plt.hist(audit_results['vsScore'], range=(-1,1), alpha=0.6, label='vaderSentiment')
plt.hist(audit_results['perspectiveScore_normalized'], range=(-1,1), alpha=0.6, label='Perspective API')

plt.xlabel('Sentiment Scores')
plt.ylabel('Frequency')
plt.title("Model Score Distributions")
plt.legend()


# ## Statistical Analysis

# In[80]:


# Create sub-dataframes for statistical analysis
black_woman = audit_results[(audit_results['Gender'] == 'woman') & (audit_results['Race'] == 'black')]
black_man = audit_results[(audit_results['Gender'] == 'man') & (audit_results['Race'] == 'black')]
white_woman = audit_results[(audit_results['Gender'] == 'woman') & (audit_results['Race'] == 'white')]
white_man = audit_results[(audit_results['Gender'] == 'man') & (audit_results['Race'] == 'white')]
asian_woman = audit_results[(audit_results['Gender'] == 'woman') & (audit_results['Race'] == 'asian')]
asian_man = audit_results[(audit_results['Gender'] == 'man') & (audit_results['Race'] == 'asian')]

man = audit_results[(audit_results['Gender'] == 'man')]
woman = audit_results[(audit_results['Gender'] == 'woman')]
asian = audit_results[(audit_results['Race'] == 'asian')]
white = audit_results[(audit_results['Race'] == 'white')]
black = audit_results[(audit_results['Race'] == 'black')]


# **Null Hypothesis**: There is no difference in mean scores among models textblob, vaderSentiment, and Google's Perspective API.
# 
# **Alternative Hypothesis**: There is a difference in mean scores among models textblob, vaderSentiment, and Google's Perspective API.
# 
# **α**: 0.05

# In[81]:


# Performs one-way ANOVA between model scores
one_way_result = f_oneway(audit_results['perspectiveScore_normalized'],
                          audit_results['tbPolairty'],
                          audit_results['vsScore'])

print("One-way ANOVA:")
print("F-statistic:", one_way_result.statistic)
print("p-value:", one_way_result.pvalue)


# **Interpretation**:
# 
# - We reject the hypothesis that there is no difference in mean scores among extblob, vaderSentiment, and Google's Perspective API.

# ### Perspective API

# **Null Hypothesis**: There is no difference in mean Perspective API scores among race and gender.
# 
# **Alternative Hypothesis**: There is a difference in mean Perspective API scores among race and gender.
# 
# **α**: 0.05

# In[82]:


# Performs two-way ANOVA on race and gender scores for Perspective API
formula = 'perspectiveScore ~ Race + Gender'
model = ols(formula, audit_results).fit()
two_way_result = anova_lm(model)

print("\nTwo-way ANOVA:")
print(two_way_result)


# In[83]:


# Performs two-way ANOVA on race scores for Perspective API
formula = 'perspectiveScore ~ Race'
model = ols(formula, audit_results).fit()
two_way_result = anova_lm(model)

print("\nTwo-way ANOVA:")
print(two_way_result)


# In[84]:


# Performs two-way ANOVA on gender scores for Perspective API
formula = 'perspectiveScore ~ Gender'
model = ols(formula, audit_results).fit()
two_way_result = anova_lm(model)

print("\nTwo-way ANOVA:")
print(two_way_result)


# In[85]:


# Performs one-way ANOVA one black and white scores for Perspective API
one_way_result = f_oneway(black['perspectiveScore'],
                          white['perspectiveScore'])

print("One-way ANOVA:")
print(one_way_result)


# In[86]:


# Performs one-way ANOVA one black and white scores for Perspective Api
one_way_result = f_oneway(asian['perspectiveScore'],
                          white['perspectiveScore'])

print("One-way ANOVA:")
print(one_way_result)


# In[87]:


# Performs one-way ANOVA one black and white scores for Perspective Api
one_way_result = f_oneway(black['perspectiveScore'],
                          white['perspectiveScore'])

print("One-way ANOVA:")
print(one_way_result)


# **Interpretation**:
# 
# - We reject the hypothesis that there is no difference in mean Perspective API scores among race and gender.
# - We fail to reject the hypothesis that there is no difference in mean Perspective API scores among gender.
# - We reject the hypothesis that there is no difference in mean Perspective API scores among races.
#     - We reject the hypothesis that there is no difference in mean Perspective API scores among black and white.
#     - We reject the hypothesis that there is no difference in mean Perspective API scores among black and asian.
#     - We reject the hypothesis that there is no difference in mean Perspective API scores among asian and white.
# 
# 

# In[88]:


# Plot Perspective API score Distributions among races 
plt.hist(white['perspectiveScore_normalized'], range=(-1,1), alpha=0.6, label='white')
plt.hist(black['perspectiveScore_normalized'], range=(-1,1), alpha=0.6, label='black')
plt.hist(asian['perspectiveScore_normalized'], range=(-1,1), alpha=0.6, label='Asian')

plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title("Race Score Distributions Perspective API")
plt.legend()


# In[101]:


# Plot Perspective API score Distributions between gender
plt.hist(woman['perspectiveScore_normalized'], range=(-1,1),alpha=0.7, label='female')
plt.hist(man['perspectiveScore_normalized'], range=(-1,1), alpha=0.7, label='male')

plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title("Gender Score Distributions Perspective API")
plt.legend()


# In[102]:


# Plot Perspective API score Distributions between gender/race identities
plt.hist(black_woman['perspectiveScore_normalized'], range=(-1,1),alpha=0.5, label='black female')
plt.hist(black_man['perspectiveScore_normalized'], range=(-1,1), alpha=0.5, label='black male')
plt.hist(white_woman['perspectiveScore_normalized'], range=(-1,1),alpha=0.5, label='white female')
plt.hist(white_man['perspectiveScore_normalized'], range=(-1,1), alpha=0.5, label='white male')
plt.hist(asian_woman['perspectiveScore_normalized'], range=(-1,1),alpha=0.5, label='asian female')
plt.hist(asian_man['perspectiveScore_normalized'], range=(-1,1), alpha=0.5, label='asian male')

plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title("Race/Gender Distributions Perspective API")
plt.legend()


# ### textblob

# **Null Hypothesis**: There is no difference in mean textblob scores among race and gender.
# 
# **Alternative Hypothesis**: There is a difference in mean textblob scores among race and gender.
# 
# **α**: 0.05

# In[22]:


# Performs two-way ANOVA on race and gender scores for textblob
formula = 'tbPolairty ~ Race + Gender'
model = ols(formula, audit_results).fit()
two_way_result = anova_lm(model)

print("\nTwo-way ANOVA:")
print(two_way_result)


# In[23]:


# Performs two-way ANOVA on race scores for textblob
formula = 'tbPolairty ~ Race'
model = ols(formula, audit_results).fit()
two_way_result = anova_lm(model)

print("\nTwo-way ANOVA:")
print(two_way_result)


# In[24]:


# Performs two-way ANOVA on race scores for textblob
formula = 'tbPolairty ~ Gender'
model = ols(formula, audit_results).fit()
two_way_result = anova_lm(model)

print("\nTwo-way ANOVA:")
print(two_way_result)


# In[95]:


# Performs one-way ANOVA on black and white scores for Perspective Api
one_way_result = f_oneway(black['tbPolairty'],
                          white['tbPolairty'])

print("One-way ANOVA:")
print(one_way_result)


# In[96]:


# Performs one-way ANOVA on black and asian scores for Perspective Api
one_way_result = f_oneway(black['tbPolairty'],
                          asian['tbPolairty'])

print("One-way ANOVA:")
print(one_way_result)


# In[97]:


# Performs one-way ANOVA on asian and white scores for Perspective Api
one_way_result = f_oneway(audit_results[audit_results['Race'] == 'white']['tbPolairty'],
                          audit_results[audit_results['Race'] == 'asian']['tbPolairty'])

print("One-way ANOVA:")
print(one_way_result)


# **Interpretation**:
# 
# - We reject the hypothesis that there is no difference in mean textblob scores among race and gender.
# - We fail to reject the hypothesis that there is no difference in mean textblob scores among gender.
# - We reject the hypothesis that there is no difference in mean textblob scores among races.
#     - We reject the hypothesis that there is no difference in mean textblob scores among black and white.
#     - We reject the hypothesis that there is no difference in mean textblob scores among black and asian.
#     - We fail to reject the hypothesis that there is no difference in mean textblob scores among asian and white.
# 
# 

# In[103]:


# Plot textblob score Distributions among races 
plt.hist(white['tbPolairty'], range=(-1,1), alpha=0.6, label='white')
plt.hist(black['tbPolairty'], range=(-1,1), alpha=0.6, label='black')
plt.hist(asian['tbPolairty'], range=(-1,1), alpha=0.6, label='Asian')

plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title("Race Score Distributions tbPolairty")
plt.legend()


# In[106]:


# Plot textblob score Distributions among races 
plt.hist(woman['tbPolairty'], range=(-1,1), alpha=0.6, label='female')
plt.hist(man['tbPolairty'], range=(-1,1), alpha=0.6, label='male')

plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title("Gender Score Distributions tbPolairty")
plt.legend()


# In[111]:


# Plot Perspective API score Distributions between gender/race identities
plt.hist(black_woman['tbPolairty'], range=(-1,1),alpha=0.5, label='black female')
plt.hist(black_man['tbPolairty'], range=(-1,1), alpha=0.5, label='black male')
plt.hist(white_woman['tbPolairty'], range=(-1,1),alpha=0.5, label='white female')
plt.hist(white_man['tbPolairty'], range=(-1,1), alpha=0.5, label='white male')
plt.hist(asian_woman['tbPolairty'], range=(-1,1),alpha=0.5, label='asian female')
plt.hist(asian_man['tbPolairty'], range=(-1,1), alpha=0.5, label='asian male')

plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title("Race/Gender Distributions tbPolairty")
plt.legend()


# ### vaderSentiment

# **Null Hypothesis**: There is no difference in mean vaderSentiment scores among race and gender.
# 
# **Alternative Hypothesis**: There is a difference in mean vaderSentiment scores among race and gender.
# 
# **α**: 0.05

# In[112]:


# Performs two-way ANOVA on race and gender scores for vaderSentiment
formula = 'vsScore ~ Race + Gender'
model = ols(formula, audit_results).fit()
two_way_result = anova_lm(model)

print("\nTwo-way ANOVA:")
print(two_way_result)


# In[113]:


# Performs one-way ANOVA on race and gender scores for vaderSentiment
formula = 'vsScore ~ Race'
model = ols(formula, audit_results).fit()
two_way_result = anova_lm(model)

print("\nTwo-way ANOVA:")
print(two_way_result)


# In[114]:


# Performs one-way ANOVA on gender scores for vaderSentiment
formula = 'vsScore ~ Gender'
model = ols(formula, audit_results).fit()
two_way_result = anova_lm(model)

print("\nTwo-way ANOVA:")
print(two_way_result)


# Interpretation:
# 
# - We fail to reject the hypothesis that there is no difference in mean textblob scores among race and gender.
# - We fail to reject the hypothesis that there is no difference in mean textblob scores among gender.
# - We fail to reject the hypothesis that there is no difference in mean textblob scores among races.

# In[119]:


# Plot vaderSentiment score Distributions among races 
plt.hist(white['vsScore'], range=(-1,1), alpha=0.6, label='white')
plt.hist(black['vsScore'], range=(-1,1), alpha=0.6, label='black')
plt.hist(asian['vsScore'], range=(-1,1), alpha=0.6, label='Asian')

plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title("Race Score Distributions vsScore")
plt.legend()


# In[121]:


# Plot vaderSentiment score Distributions among races 
plt.hist(woman['vsScore'], range=(-1,1), alpha=0.6, label='female')
plt.hist(man['vsScore'], range=(-1,1), alpha=0.6, label='male')

plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title("Gender Score Distributions vsScore")
plt.legend()


# In[ ]:




