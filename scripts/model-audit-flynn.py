#!/usr/bin/env python
# coding: utf-8

# # Model Exploration
# Purpose is to understand models `textblob`, `vaderSentiment`, and Google's `Perspective API`.<br>
# By: Flynn O'Sullivan (derived from Elsie's notebook)<br>
# Date: 2/21/23

# In[1]:


# Model Imports
from modelCollection import ModelCollection

# Overhead Imports
from json import load

import pandas as pd
import spacy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import f_oneway
import matplotlib.patheffects as pe


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
    sentence = sentence.lower()
    for identity in race_gender_identities:
        identity_lower = identity.lower()
        sentence = sentence.replace(identity_lower, '[]')
        
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


def replace_race(sentence):
    races = ['Asian', 'white', 'black', 'Middle Eastern']
    sentence = sentence.lower()
    for i in races:
        race_lower = i.lower()
        sentence = sentence.replace(race_lower, '[]')
    return sentence


# In[8]:


# Clean data: change to template sentences excluding race and gender
sentences[["Sentences", "Num Pronouns"]] = sentences["Sentences"].apply(lambda x: pd.Series(replace_gender(x)))
sentences["Sentences"] = sentences["Sentences"].apply(replace_race_gender)
sentences['Sentences'] = sentences['Sentences'].apply(replace_race)


# ## Sample Testing

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
# original_sentence = sample_df['Sentences'].loc[10]
# identity = "Middle Eastern woman"
# modified = fill_race_gender(original_sentence, identity)

# print('Original:\n', original_sentence)
# print('Modified:\n', modified)


# In[11]:


# Initialize Model Collection and testing sentances
mc = ModelCollection(gcp_api_key=SECRETS['PerspectiveAPIKey'])


# In[12]:


races = ['asian', 'black', 'white']
gender = ['him', 'her', 'man', 'woman']


# In[13]:


add_columns = ['baseline_perspective', 'baseline_tb_polarity', 'baseline_tb_obj', 'baseline_vader',
              'asian_woman_perspective', 'asian_woman_tb_polarity', 'asian_woman_tb_obj', 'asian_woman_vader',
              'asian_man_perspective', 'asian_man_tb_polarity', 'asian_man_tb_obj', 'asian_man_vader',
              'white_woman_perspective', 'white_woman_tb_polarity', 'white_woman_tb_obj', 'white_woman_vader',
              'white_man_perspective', 'white_man_tb_polarity', 'white_man_tb_obj', 'white_man_vader',
              'black_woman_perspective', 'black_woman_tb_polarity', 'black_woman_tb_obj', 'black_woman_vader',
              'black_man_perspective', 'black_man_tb_polarity', 'black_man_tb_obj', 'black_man_vader',
              'middle_eastern_man_perspective', 'middle_eastern_man_tb_polarity', 'middle_eastern_man_tb_obj', 'middle_eastern_man_vader',
              'middle_eastern_woman_perspective', 'middle_eastern_woman_tb_polarity', 'middle_eastern_woman_tb_obj', 'middle_eastern_woman_vader']
for i in add_columns:
    sentences[i] = pd.NA


# In[14]:


sentences


# In[15]:


results_df = pd.DataFrame(columns=['Sentiment', 'Sentence', 'Num_Pronouns', 'Race', 'Gender', 'Model', 'Score'])
to_concat = []
for idx, entry in tqdm(sentences.iterrows()):
    race_gender_combos = ['Asian woman', 'Asian man', 'White woman', 'White man', 'Black woman', 'Black man',
                          'Middle Eastern woman', 'Middle Eastern man']
    models = ['tb_polarity', 'tb_objectivity', 'perspective', 'vader']
    for i in race_gender_combos:
        append_dict = {}
        append_dict['Sentiment'] = entry['Sentiment']
        append_dict['Sentence'] = entry['Sentences']
        append_dict['Num_Pronouns'] = entry['Num Pronouns']
        combo = i.lower().split()
        if len(combo) == 3:
            race = combo[0] +' '+ combo[1]
        elif len(combo) == 2:
            race = combo[0]
        gender = combo[-1]
        append_dict['Race'] = race
        append_dict['Gender'] = gender
        mod_sentence = fill_race_gender(entry['Sentences'], i)
        for j in models:
            append_dict['Model'] = j
            if j == 'tb_polarity':
                score = mc.queryTextBlobPolarity(mod_sentence)
                append_dict['Score'] = score
            if j == 'tb_objectivity':
                score = mc.queryTextBlobObjectivity(mod_sentence)
            if j == 'perspective':
                score = mc.queryPerspective(mod_sentence)
            if j == 'vader':
                score = mc.queryVaderSentiment(mod_sentence)
            append_dict['Score'] = score
            append_df = pd.DataFrame(append_dict, index=[0])
            to_concat.append(append_df)


# In[16]:


df = pd.concat(to_concat, ignore_index=True)


# In[18]:


for idx, entry in tqdm(sentences.iterrows()):
    race_gender_combos = ['Asian woman', 'Asian man', 'White woman', 'White man', 'Black woman', 'Black man',
                          'Middle Eastern woman', 'Middle Eastern man']
    sentiment = entry['Sentiment']
    sentence = entry['Sentences']
    num_pronouns = entry['Num Pronouns']
    race = 'baseline'
    gender = 'baseline'
    models = ['tb_polarity', 'tb_objectivity', 'perspective', 'vader']
    for j in models:
        model = j
        if j == 'tb_polarity':
            score = mc.queryTextBlobPolarity(sentence)
            append_dict['Score'] = score
        if j == 'tb_objectivity':
            score = mc.queryTextBlobObjectivity(sentence)
        if j == 'perspective':
            score = mc.queryPerspective(sentence)
        if j == 'vader':
            score = mc.queryVaderSentiment(sentence)
        df.loc[len(df.index)] = [sentiment, sentence, num_pronouns, race, gender, model, score]


# In[19]:


df


# In[20]:


df['race_gender_combo'] = df.apply(lambda x: x['Race'] + ' '+ x['Gender'], axis=1)


# In[21]:


tp = df[df['Model'] == 'tb_polarity'].groupby('race_gender_combo')['Score'].mean() - df[df['Model'] == 'tb_polarity'].groupby('race_gender_combo')['Score'].mean()['baseline baseline']
to = df[df['Model'] == 'tb_objectivity'].groupby('race_gender_combo')['Score'].mean() - df[df['Model'] == 'tb_objectivity'].groupby('race_gender_combo')['Score'].mean()['baseline baseline']
p = df[df['Model'] == 'perspective'].groupby('race_gender_combo')['Score'].mean() - df[df['Model'] == 'perspective'].groupby('race_gender_combo')['Score'].mean()['baseline baseline']
race_gender_df = pd.concat([tp, to, p],axis=1)
race_gender_df.columns = ['tp_polarity', 'tb_objectivity', 'perspective']
race_gender_df = race_gender_df.drop('baseline baseline', axis='index')


# In[22]:


race_tp = df[df['Model'] == 'tb_polarity'].groupby('Race')['Score'].mean() - df[df['Model'] == 'tb_polarity'].groupby('Race')['Score'].mean()['baseline']
race_to = df[df['Model'] == 'tb_objectivity'].groupby('Race')['Score'].mean() - df[df['Model'] == 'tb_objectivity'].groupby('Race')['Score'].mean()['baseline']
race_perspective = df[df['Model'] == 'perspective'].groupby('Race')['Score'].mean() - df[df['Model'] == 'perspective'].groupby('Race')['Score'].mean()['baseline']
gender_tp = df[df['Model'] == 'tb_polarity'].groupby('Gender')['Score'].mean() - df[df['Model'] == 'tb_polarity'].groupby('Gender')['Score'].mean()['baseline']
gender_to = df[df['Model'] == 'tb_objectivity'].groupby('Gender')['Score'].mean() - df[df['Model'] == 'tb_objectivity'].groupby('Gender')['Score'].mean()['baseline']
gender_perspective = df[df['Model'] == 'perspective'].groupby('Gender')['Score'].mean() - df[df['Model'] == 'perspective'].groupby('Gender')['Score'].mean()['baseline']

race_df = pd.concat([race_tp, race_to, race_perspective], axis=1)
race_df.columns = ['tb_polarity', 'tb_objectivity', 'perspective']
race_df = race_df.drop('baseline', axis='index')

gender_df = pd.concat([gender_tp, gender_to, gender_perspective], axis=1)
gender_df.columns = ['tb_polarity', 'tb_objectivity', 'perspective']
gender_df = gender_df.drop('baseline', axis='index')


# In[23]:


race_gender_df


# In[24]:


gender_df


# In[25]:


race_df


# In[26]:


race_gender_df.columns = ['Perspective API', 'TextBlob Objectivity', 'TextBlob Polarity']
race_gender_df.index = race_gender_df.index.str.capitalize()
race_gender_df.index


# In[27]:


race_gender_df


# In[28]:


def plot_matrix(df, title, blurbs=['A positive value in this row indicates the sentence with the identifier is more toxic than the same sentence with no identifiers',
                                                        'A positive value in this row indicates the sentence with the identifier is more objective than the same sentence with no identifiers',
                                                        'A positive value in this row indicates the sentence with the identifier is more polarizing than the same sentence with no identifiers']):
    cmap = LinearSegmentedColormap.from_list('rg', ['red', 'white', 'green'], N=256)
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(df.T, annot=True, fmt='.6f', cmap=cmap, cbar=False)
    ax.set_title(title)
    ax.set_xlabel('Identifiers')
    ax.set_ylabel('Model Audited')
    plt.xticks(rotation=25)
    plt.yticks(rotation=0)
    for i, blurb in enumerate(blurbs):
        ax.text(4, i + 0.8, blurb, ha='center', va='center', fontsize='7.95', color='black', path_effects=[pe.withStroke(linewidth=2.5, foreground='white')], transform=ax.transData)
    plt.tight_layout()
    plt.show()


# In[29]:


plot_matrix(race_gender_df, 'Race-Gender Bias Matrix')


# In[30]:


plot_matrix(race_df, 'Race Bias Matrix')


# In[31]:


# Querying using individual methods
results = []
for idx, entry in tqdm(sentences.iterrows()):
    sentence = entry['Sentences']

    baseline_tb_pol = mc.queryTextBlobPolarity(sentence)
    baseline_perspective = mc.queryPerspective(sentence)
    baseline_tb_obj = mc.queryTextBlobObjectivity(sentence)
    baseline_vader = mc.queryVaderSentiment(sentence)
    sentences.loc[idx, 'baseline_tb_polarity'] = baseline_tb_pol
    sentences.loc[idx, 'baseline_perspective'] = baseline_perspective
    sentences.loc[idx, 'baseline_tb_obj'] = baseline_tb_obj
    sentences.loc[idx, 'baseline_vader'] = baseline_vader
    
    race_gender_combos = ['Asian woman', 'Asian man', 'White woman', 'White man', 'Black woman', 'Black man',
                         'Middle Eastern woman', 'Middle Eastern man']
    
    for identity in race_gender_combos:
        modified_sentence = fill_race_gender(sentence, identity)
        mod_perspective = mc.queryPerspective(modified_sentence)
        mod_tb_obj = mc.queryTextBlobObjectivity(modified_sentence)
        mod_tb_pol = mc.queryTextBlobPolarity(modified_sentence)
        mod_vader = mc.queryVaderSentiment(modified_sentence)
        
        col_names = '_'.join(identity.lower().split())
        sentences.loc[idx, col_names+'_tb_polarity'] = mod_tb_pol
        sentences.loc[idx, col_names+'_tb_obj'] = mod_tb_obj
        sentences.loc[idx, col_names+'_perspective'] = mod_perspective
        sentences.loc[idx, col_names+'_vader'] = mod_vader
    

#     result = {
#         'perspectiveScore': mc.queryPerspective(modified_sentence),
#         'tbPolairty': mc.queryTextBlobPolairty(modified_sentence),
#         'tbObjectivity': mc.queryTextBlobObjectivity(modified_sentence),
#         'vsScore': mc.queryVaderSentiment(modified_sentence)
#     }
#     results.append(result)
    


# In[32]:


race_gender_combos = ['Asian woman', 'Asian man', 'White woman', 'White man', 'Black woman', 'Black man',
                      'Middle Eastern woman', 'Middle Eastern man']
for i in race_gender_combos:
    s = i.lower().split()
    col_name = ''
    col_name2 = '_'.join(s)
    for j in s:
        col_name += j[0].lower()
    sentences[col_name+'_perspective_diff'] = sentences[col_name2+'_perspective'] - sentences['baseline_perspective']
    sentences[col_name+'_tb_obj_diff'] = sentences[col_name2+'_tb_obj'] - sentences['baseline_tb_obj']
    sentences[col_name+'_tb_pol_diff'] = sentences[col_name2+'_tb_polarity'] - sentences['baseline_tb_polarity']
    sentences[col_name+'_vader_diff'] = sentences[col_name2+'_vader'] - sentences['baseline_vader']


# In[34]:


column_averages = {}
for column in sentences:
    if 'diff' in column:
        column_averages[column] = sentences[column].mean()


# In[35]:


column_averages


# In[37]:


gender_vis = {}
vader_woman = (df[df['Model'] == 'vader'].groupby('Gender').Score.mean() - df[df['Model'] == 'vader'].groupby('Gender').Score.mean()['baseline']).drop('baseline', axis='index')['woman']
vader_man = (df[df['Model'] == 'vader'].groupby('Gender').Score.mean() - df[df['Model'] == 'vader'].groupby('Gender').Score.mean()['baseline']).drop('baseline', axis='index')['man']
gender_vis['vader_man'] = vader_man
gender_vis['vader_woman'] = vader_woman

tb_obj_man = (df[df['Model'] == 'tb_objectivity'].groupby('Gender').Score.mean() - df[df['Model'] == 'tb_objectivity'].groupby('Gender').Score.mean()['baseline']).drop('baseline', axis='index')['man']
tb_obj_woman = (df[df['Model'] == 'tb_objectivity'].groupby('Gender').Score.mean() - df[df['Model'] == 'tb_objectivity'].groupby('Gender').Score.mean()['baseline']).drop('baseline', axis='index')['woman']
gender_vis['tb_obj_man'] = tb_obj_man
gender_vis['tb_obj_woman'] = tb_obj_woman

tb_pol_man = (df[df['Model'] == 'tb_polarity'].groupby('Gender').Score.mean() - df[df['Model'] == 'tb_polarity'].groupby('Gender').Score.mean()['baseline']).drop('baseline', axis='index')['man']
tb_pol_woman = (df[df['Model'] == 'tb_polarity'].groupby('Gender').Score.mean() - df[df['Model'] == 'tb_polarity'].groupby('Gender').Score.mean()['baseline']).drop('baseline', axis='index')['woman']
gender_vis['tb_pol_man'] = tb_pol_man
gender_vis['tb_pol_woman'] = tb_pol_woman

perspective_man = (df[df['Model'] == 'perspective'].groupby('Gender').Score.mean() - df[df['Model'] == 'perspective'].groupby('Gender').Score.mean()['baseline']).drop('baseline', axis='index')['man']
perspective_woman = (df[df['Model'] == 'perspective'].groupby('Gender').Score.mean() - df[df['Model'] == 'perspective'].groupby('Gender').Score.mean()['baseline']).drop('baseline', axis='index')['man']
gender_vis['perspective_man'] = perspective_man
gender_vis['perspective_woman'] = perspective_woman



gender_vis


# In[38]:


column_averages


# In[39]:


grouped_data = {}
for key, value in column_averages.items():
    group, *metric_parts = key.split('_')
    metric = '_'.join(metric_parts)
    if group not in grouped_data:
        grouped_data[group] = {}
    grouped_data[group][metric] = value
labels = sorted(set('_'.join(key.split('_')[1:]) for key in column_averages.keys()))
all_metrics = sorted(set(metric for group_metrics in grouped_data.values() for metric in group_metrics))
for group_metrics in grouped_data.values():
    for metric in all_metrics:
        if metric not in group_metrics:
            group_metrics[metric] = 0
group_names = ['Asian woman', 'Asian man', 'White woman', 'White man', 'Black woman', 'Black man', 'Middle Eastern woman', 'Middle Eastern man']
values = np.array([[group_metrics[metric] for metric in all_metrics] for group_metrics in grouped_data.values()])
fig, ax = plt.subplots(figsize=(10,8))
index = np.arange(len(all_metrics))
bar_width = 0.1
opacity = 0.8
for i, (group_name, group_values) in enumerate(zip(group_names, values)):
    plt.bar(index + i*bar_width, group_values, bar_width, alpha=opacity, label=group_name)
plt.xlabel('Model Audited')
plt.ylabel('Difference from baseline score')
plt.title('Difference in Model Scoring Between Sentence with Identifier and Sentence Without')
plt.xticks(index + bar_width * (len(group_names) / 2), all_metrics, rotation=0)
ax.set_xticklabels(['Perspective API', 'TextBlob Objectivity', 'TextBlob Polarity', 'VaderSentiment'])
plt.legend()
plt.tight_layout()
plt.show()


# In[40]:


grouped = sentences.groupby('Sentiment')
mean_diffs = grouped[['aw_perspective_diff', 'aw_tb_obj_diff', 'aw_tb_pol_diff',
       'aw_vader_diff', 'am_perspective_diff', 'am_tb_obj_diff',
       'am_tb_pol_diff', 'am_vader_diff', 'ww_perspective_diff',
       'ww_tb_obj_diff', 'ww_tb_pol_diff', 'ww_vader_diff',
       'wm_perspective_diff', 'wm_tb_obj_diff', 'wm_tb_pol_diff',
       'wm_vader_diff', 'bw_perspective_diff', 'bw_tb_obj_diff',
       'bw_tb_pol_diff', 'bw_vader_diff', 'bm_perspective_diff',
       'bm_tb_obj_diff', 'bm_tb_pol_diff', 'bm_vader_diff',
       'mew_perspective_diff', 'mew_tb_obj_diff', 'mew_tb_pol_diff',
       'mew_vader_diff', 'mem_perspective_diff', 'mem_tb_obj_diff',
       'mem_tb_pol_diff', 'mem_vader_diff']].mean()


# In[41]:


df.groupby('race_gender_combo').Score.mean()


# In[42]:


cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'red'), (0.5, 'white'), (1, 'green')])


# In[43]:


mean_diffs


# In[44]:


max_abs_value = mean_diffs.abs().max().max()
normalized = mean_diffs / max_abs_value
normalized = normalized.apply(pd.to_numeric, errors='coerce')
mean_diffs = mean_diffs.apply(pd.to_numeric, errors='coerce')


# In[45]:


mean_diffs


# In[46]:


plt.figure(figsize=(20,6))
sns.heatmap(mean_diffs, annot=True, fmt=".2f", cmap=cmap, cbar=False,
           xticklabels=mean_diffs.columns, yticklabels=mean_diffs.index)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Heatmap of Sentiment Differences')
plt.tight_layout()
plt.show()


# # Perspective

# In[47]:


diff_cols = [col for col in sentences.columns if 'perspective_diff' in col]
diff_values = [sentences[col].values for col in diff_cols]
f_statistic, p_value = f_oneway(*diff_values)
p_value


# # Text Blob (polarity)

# In[48]:


diff_cols = [col for col in sentences.columns if 'tb_pol_diff' in col]
diff_values = [sentences[col].values for col in diff_cols]
f_statistic, p_value = f_oneway(*diff_values)
p_value


# # Text Blob (objectivity)

# In[49]:


diff_cols = [col for col in sentences.columns if 'tb_obj_diff' in col]
diff_values = [sentences[col].values for col in diff_cols]
f_statistic, p_value = f_oneway(*diff_values)
p_value

