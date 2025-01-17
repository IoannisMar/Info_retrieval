import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import string
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.probability import FreqDist

# Προετοιμασία stopwords και σημεία στίξης
stop_words = set(stopwords.words('english'))
punctuation_table = str.maketrans('', '', string.punctuation)


pages = ["World_Wide_Web", "Web_Development", "Search_Engines", "Social_Media"]# Λίστα με τα ονόματα των άρθρων που θα λάβουμε τις πληροφορίες παρακάτω


data = []# Αρχικοποίηση λίστας για την αποθήκευση των δεδομένων

#υλοποίηση web crawler
for page_id, page in enumerate(pages):  #Αντί για enumerate(pages), ξεκινάμε από το 0
 url = f"https://en.wikipedia.org/wiki/{page}"
 try:
  response = requests.get(url)#Αποστολή αιτήματος στην ιστοσελίδα
  response.raise_for_status()
  soup = BeautifulSoup(response.text, 'html.parser')#εφαρμογή BeautifulSoup στο κείμενο που λάβαμε

  paragraphs = soup.find_all('p')#Εξαγωγή παραγράφων από το html
  parsed_paragraphs = [p.text.strip() for p in paragraphs if p.text.strip()]

  for paragraph in parsed_paragraphs:
   data.append({'Selida': page, 'Selida_ID': page_id, 'Paragraphos': paragraph})#Προσθήκη δεδομένων στη λίστα data
    
 except requests.RequestException as e:
  print(f"Error sthn ektelesh tou etoimatos {url}: {e}")#exception για την περίπτωση που δεν λάβουμε απάντηση στο αίτημα
        
df = pd.DataFrame(data)# Δημιουργία DataFrame


def clean_and_tokenize(text):
 tokens = word_tokenize(text.lower())#Μετατροπή των γραμμάτων σε πεζά και ταυτόχρονο tokenization
 cleaned_tokens = [token.translate(punctuation_table) for token in tokens if token not in stop_words and token.isalnum()]#αφαίρεση "stop_words" και σημείων στίξης"
 return cleaned_tokens

df['Paragraphos'] = df['Paragraphos'].apply(clean_and_tokenize)#κλήση του υποπρογράμματος για προεπεξεργασία του κειμένου

all_tokens = [token for tokens in df['Paragraphos'] for token in tokens]#Δημιουργία λίστα με όλες τις λέξεις από την στήλη Paragraphos
fdist1 = FreqDist(all_tokens)#Δημιουργία αντικειμένου που περιέχει την συχνότητα εμφάνισης λέξεων στα άρθρα
print("Top 50 Most Common Tokens:")
print(fdist1.most_common(50))#Εμφάνιση των 50 συχνότερων λέξεων στην κονσόλα
fdist1.plot(50, title='Top 50 Most Common Tokens')#Εμφάνιση γραφικής παράστασης με τις 50 συχνότερες λέξεις

inverted_index = defaultdict(set)  # Δημιουργία Ανεστραμμένου Ευρετηρίου

for _, row in df.iterrows():#Για κάθε σειρά του df(το _ σημαίνει πως δεν μας ενδιαφέρει ο index που επιστρέφει η συνάρτηση iterrows())
 page_id = row['Selida_ID']  #χρήση του ID της σελίδας
 for token in row['Paragraphos']:#για κάθε token 
  inverted_index[token].add(page_id)  #προσθήκη του ID της σελίδας στο set

inverted_index_csv = [{"Term": term, "Document IDs": ",".join(map(str, sorted(doc_ids)))} for term, doc_ids in inverted_index.items()]#μετατροπή του ευρετηρίου σε λίστα ώστε να γίνει μετά DataFrame και στην συνέχεια να αποθηκευτεί σε CSV αρχείο
inverted_index_df = pd.DataFrame(inverted_index_csv)# Δημιουργία του DataFrame ευρετηρίου για αποθήκευση σε μορφή CSV παρακάτω
inverted_index_df.to_csv('inverted_index.csv', index=False, encoding='utf-8')#αποθήκευση σε μορφή csv

df.to_csv('ApotelesmataXarakthrwn.csv', index=False, encoding='utf-8')#αποθήκευση του "καθαρισμένου" DataFrame σε αρχείο CSV

inverted_index = pd.read_csv('inverted_index.csv')
df = pd.read_csv('ApotelesmataXarakthrwn.csv')# Φόρτωση των έτοιμων δεδομένων για την μηχανή αναζήτησης

# Προετοιμασία δεδομένων για TF-IDF και VSM
corpus = df['Paragraphos'].apply(lambda x: ' '.join(eval(x))).tolist()#δημιουργία της λίστας corpus η οποία θα περιέχει ενιαία strings αντι για λίστες με tokens
vectorizer = TfidfVectorizer()#αρχικοποίηση ενός TfidfVectorizer 
tfidf_matrix = vectorizer.fit_transform(corpus)#μετατροπή του corpus σε πίνακα Tfidf

def search_query(query, algorithm):#Συνάρτηση search_query για την υλοποίηση των αναζητήσεων
 query_tokens = word_tokenize(query.lower())#Query_tokens περιλαμβάνει τα tokens των αρχείων
 if algorithm == "Boolean Retrieval":#ΠΕΡΙΠΤΩΣΗ 1: Boolean Retrieval
  if "and" in query_tokens:#ΠΕΡΙΠΤΩΣΗ 1.1: Επιλογή AND
   terms = query.split('AND')
   terms = [term.strip() for term in terms]
            
   doc_sets = []#Ανακτάμε τα έγγραφα για την συγκρίσεις και την εύρεση κοινών
   for term in terms:#Για τον αριθμό των terms υλοποιούμε επανάληψη for
    result = inverted_index[inverted_index['Term'].isin(word_tokenize(term.lower()))]#Θέτουμε την εντολή result
    if not result.empty:#Σε περίπτωση που το result δεν είναι κενό
     document_ids = set()#Δημιουργία κενού συνόλου στο document_ids
     for term in word_tokenize(term.lower()):#Για term διαχωρίζει την λέξη και διασφαλίζει την περίπτωση case-insensitive
      docs = result[result['Term'] == term]['Document IDs'].values#Πραγματοποίηση της αναζήτησης στο ευρετήριο και επιστροφή των IDs των αρχείων που βρέθηκε
      if docs.size > 0:#Έλεγχος αν βρέθηκε σε τουλάχιστον 1 έγγραφο
       doc_ids = set(map(int, docs[0].split(',')))#Δημιουργία map τύπου int για τα doc_ids
       document_ids.update(doc_ids)#Προσθήκη IDs στο document_ids
     doc_sets.append(document_ids)#Προσθήκη του συνόλου των εγγράφων document_ids στο doc_sets

   common_docs = set.intersection(*doc_sets)#Δημιουργία common_docs στην οποία εισχωρούμε τα έγγραφα που περιέχουν και τις 2 λέξεις
   if not common_docs:#Περίπτωση που δεν υπάρχουν τέτοια αρχεία
    return [f"Δεν βρέθηκαν αρχεία που να περιέχουν και τις δύο λέξεις."]
            
   results = [f"Αρχείο: {df[df['Selida_ID'] == doc_id]['Selida'].values[0]}" for doc_id in common_docs]#Εμφάνιση μηνύματος και των εγγράφων που βρέθηκαν οι λέξεις
   return results if results else [f"Δεν βρέθηκαν αρχεία που να περιέχουν και τις δύο λέξεις."]
        
  elif "or" in query_tokens:#ΠΕΡΊΠΤΩΣΗ 1.2: Επιλογή OR
   terms = query.split('OR')
   terms = [term.strip() for term in terms]
            
   document_ids = set()#Δημιουργία κενού συνόλου στο document_ids
   for term in terms:
    result = inverted_index[inverted_index['Term'].isin(word_tokenize(term.lower()))]#Θέτουμε την εντολή result
    if not result.empty:#Σε περίπτωση που το result δεν είναι κενό
     for term in word_tokenize(term.lower()):#Για term διαχωρίζει την λέξη και διασφαλίζει την περίπτωση case-insensitive
      docs = result[result['Term'] == term]['Document IDs'].values#Πραγματοποίηση της αναζήτησης στο ευρετήριο και επιστροφή των IDs των αρχείων που βρέθηκε
      if docs.size > 0:#Έλεγχος αν βρέθηκε σε τουλάχιστον 1 έγγραφο
       doc_ids = set(map(int, docs[0].split(',')))#Δημιουργία map τύπου int για τα doc_ids
       document_ids.update(doc_ids)#Προσθήκη IDs στο document_ids

   results = [f"Αρχείο: {df[df['Selida_ID'] == doc_id]['Selida'].values[0]}" for doc_id in document_ids]#Εμφάνιση μηνύματος και των εγγράφων που βρέθηκαν οι λέξεις
   return results if results else [f"Δεν βρέθηκαν αρχεία που να περιέχουν κάποια από τις λέξεις."]#Εμφάνιση σε περίπτωση που δεν υπήρχαν αρχεία
        
  elif "not" in query_tokens:#ΠΕΡΙΠΤΩΣΗ 1.3: Επιλογή NOΤ
   terms = query.split('NOT')
   terms = [term.strip() for term in terms]
   result_include = inverted_index[inverted_index['Term'].isin(word_tokenize(terms[0].lower()))]#Για την 1η λέξη του χρήστη
   result_exclude = inverted_index[inverted_index['Term'].isin(word_tokenize(terms[1].lower()))]#Για την 2η λέξη του χρήστη(Δεξιά της NOT, για να παραλειφθεί)
   if not result_include.empty:#Σε περίπτωση που το result_include δεν είναι κενό
    included_doc_ids = set()#Δημιουργία κενού συνόλου στο included_document_ids
    for term in word_tokenize(terms[0].lower()):#Για term διαχωρίζει την λέξη και διασφαλίζει την περίπτωση case-insensitive
     docs = result_include[result_include['Term'] == term]['Document IDs'].values#Πραγματοποίηση της αναζήτησης στο ευρετήριο και επιστροφή των IDs των αρχείων που βρέθηκε
     if docs.size > 0:#Έλεγχος αν βρέθηκε σε τουλάχιστον 1 έγγραφο
      included_doc_ids.update(set(map(int, docs[0].split(','))))#Δημιουργία map τύπου int για τα included_doc_ids
            
   if not result_exclude.empty:#Σε περίπτωση που το result_exclude δεν είναι κενό
    excluded_doc_ids = set()#Δημιουργία κενού συνόλου στο excluded_document_ids
    for term in word_tokenize(terms[1].lower()):#Για term διαχωρίζει την λέξη και διασφαλίζει την περίπτωση case-insensitive
     docs = result_exclude[result_exclude['Term'] == term]['Document IDs'].values#Πραγματοποίηση της αναζήτησης στο ευρετήριο και επιστροφή των IDs των αρχείων που βρέθηκε
     if docs.size > 0:#Έλεγχος αν βρέθηκε σε τουλάχιστον 1 έγγραφο
      excluded_doc_ids.update(set(map(int, docs[0].split(','))))#Δημιουργία map τύπου int για τα excluded_doc_ids
                        
   final_results = included_doc_ids - excluded_doc_ids#Δημιουργία των final_results που περιέχουν την λέξη στα αριστερά και έχουν αποκλεισμένη της λέξη στα δεξιά
   results = [f"Αρχείο: {df[df['Selida_ID'] == doc_id]['Selida'].values[0]}" for doc_id in final_results]#Εμφάνιση μηνύματος και των εγγράφων που βρέθηκαν οι λέξεις
   return results if results else [f"Δεν βρέθηκαν αρχεία που να περιέχουν τον όρο '{terms[0]}' και να μην περιέχουν τον όρο '{terms[1]}'."]#Εμφάνιση σε περίπτωση που δεν υπήρχαν αρχεία

  else:
   result = inverted_index[inverted_index['Term'].isin(query_tokens)]#Σε περίπτωση που δεν υπάρχει λογική σχέση θα αναζητήσει μεμονωμένη λέξη
   if result.empty:#Σε περίπτωση που δεν βρέθηκε στο ευρετήριο η λέξη
    return [f"Η λέξη '{query}' δεν βρέθηκε στο ευρετήριο."]
   document_ids = set()#Δημιουργία κενού συνόλου στο document_ids
   for term in query_tokens:#Για term επανάληψη μέσα σε όλα τα tokens του ευρετηρίου
    docs = result[result['Term'] == term]['Document IDs'].values#Πραγματοποίηση της αναζήτησης στο ευρετήριο και επιστροφή των IDs των αρχείων που βρέθηκε
    if docs.size > 0:#Έλεγχος αν βρέθηκε σε τουλάχιστον 1 έγγραφο
     doc_ids = set(map(int, docs[0].split(',')))#Δημιουργία map τύπου int για τα doc_ids
     document_ids.update(doc_ids)#Προσθήκη IDs στο document_ids
            
   results = [f"Αρχείο: {df[df['Selida_ID'] == doc_id]['Selida'].values[0]}" for doc_id in document_ids]#Εμφάνιση μηνύματος και των εγγράφων που βρέθηκαν οι λέξεις
   return results if results else [f"Δεν βρέθηκαν αποτελέσματα για το '{query}'."]#Εμφάνιση σε περίπτωση που δεν υπήρχαν αρχεία

########################################################################################################################################3

 elif algorithm == "TF-IDF":
    query_vector = vectorizer.transform([query])  # Μετασχηματίζει το ερώτημα σε διανύσματα TF-IDF
    scores = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()  # Υπολογίζει τα σκορ (similarity) μεταξύ του ερωτήματος και των εγγράφων
    ranked_indices = np.argsort(scores)[::-1]  # Ταξινομεί τα σκορ κατά φθίνουσα σειρά
    results = [f"Αρχείο: {df.iloc[i]['Selida']} | Σκορ TF-IDF: {scores[i]:.4f}" for i in ranked_indices if scores[i] > 0]  # Δημιουργεί τα αποτελέσματα (ταξινομημένα) με βάση τα σκορ TF-IDF
    return results if results else [f"Δεν βρέθηκαν αποτελέσματα για '{query}' με TF-IDF."]  # Επιστρέφει τα αποτελέσματα ή μήνυμα αν δεν βρέθηκαν

########################################################################################################################################

 elif algorithm == "Vector Space Model":
    query_vector = vectorizer.transform([query])  # Μετασχηματίζει το ερώτημα σε διανύσματα TF-IDF  
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()  # Υπολογίζει την ομοιότητα (cosine similarity) μεταξύ του ερωτήματος και των εγγράφων  
    ranked_indices = np.argsort(similarities)  # Ταξινομεί τις ομοιότητες κατά αύξουσα σειρά (χωρίς το[::-1])
    results = [f"Αρχείο: {df.iloc[i]['Selida']} | Ομοιότητα: {similarities[i]:.4f}" for i in ranked_indices if similarities[i] > 0]  # Δημιουργεί τα αποτελέσματα (ταξινομημένα) με βάση την ομοιότητα
    return results if results else [f"Δεν βρέθηκαν αποτελέσματα για '{query}' με το Vector Space Model."]  # Επιστρέφει τα αποτελέσματα ή μήνυμα αν δεν βρέθηκαν

def perform_search():#Δημιουργία συνάρτησης που εκτελεί την επιθυμιτή αναζήτηση και λέξη
    query = search_entry.get().strip()#Εδώ περνάει η λέξη του χρήστη
    algorithm = algorithm_selection.get()#Αναζήτηση που θα εκτελεστεί
    #Σε περίπτωση που ο χρήστης δεν εισάγει κάποια λέξη αλλά πατήσει αναζήτηση
    if not query:
        results_text.set("Παρακαλώ εισάγετε μια λέξη για αναζήτηση.")
        return
    results = search_query(query, algorithm)#Αποτελέσματα αναζήτησης
    results_text.set("\n".join(results))#Αποθήκευση αποτελεσμάτων στην results

# Δημιουργία παραθύρου και τα χαρακτηριστικά του
root = tk.Tk()
root.title("Μηχανή Αναζήτησης")
root.geometry("800x600")
root.configure(bg="#f4f4f4")

title_label = tk.Label(#Label για τον τίτλο του interface
    root, text="Μηχανή Αναζήτησης", font=("Arial", 24, "bold"), bg="#f4f4f4", fg="#333"#Χαρακτηριστικά του τίτλου
)
title_label.pack(pady=10)#Απόσταση 10 

algorithm_selection = ttk.Combobox(root, values=["Boolean Retrieval", "TF-IDF", "Vector Space Model"], font=("Arial", 14))#Εδώ ο χρήστης επιλέγει ποια απο τις 3 αναζητήσεις επιθυμεί
algorithm_selection.set("Boolean Retrieval")#Η προεπιλογή του συστήματος θα είναι η Boolean Retrieval, σε περίπτωση που επιθυμεί να επιλέξει άλλη στο δεξί μέρος υπάρχει το βελάκι
algorithm_selection.pack(pady=10)

search_entry = ttk.Entry(root, font=("Arial", 14))#Δημιουργία του πεδίου εισαγωγής και επιλογή γραμματοσειράς
search_entry.pack(pady=10, padx=20, fill="x")#Συμπλήρωση του x με όποιο token επιθυμεί ο χρήστης

search_button = ttk.Button(root, text="Αναζήτηση", command=perform_search)#Δημιουργία κουμπιού αναζήτησης
search_button.pack(pady=5)#Τοποθέτηση του κουμπιού στην απόσταση pady=5

results_frame = tk.Frame(root, bg="#f4f4f4")#Δημιουργία πλαισίου για τα αποτελέσματα της αναζήτησης
results_frame.pack(pady=10, padx=20, fill="both", expand=True)#Συντεταγμένες του πλαισίου αποτελεσμάτων

results_text = tk.StringVar()#Καθοριμσός τύπου string του results_text
results_label = tk.Label(#Καταγράφεται στο label του results το τι θα περιλαμβάνει καθώς και τα χαρακτηριστικά του
    results_frame,
    textvariable=results_text,
    font=("Courier New", 12),
    justify="left",
    bg="#f4f4f4",
    fg="#333",
    wraplength=760,
    anchor="nw",
)
results_label.pack(fill="both", expand=True)

style = ttk.Style()
style.configure("TButton", font=("Arial", 14), padding=5) #Διακόσμηση του button του interface με απόσταση 5 από τα άλλα στοιχεία του περιβάλλοντος
style.configure("TEntry", padding=5)#Εγχώρηση απόστασης κουμπιού από την μηχανή αναζήτησης με απόσταση 5 από τα άλλα στοιχεία του περιβάλλοντος
style.configure("TCombobox", font=("Arial", 14))#Καθορισμός γραμματοσειράς 

root.mainloop() # Εκκίνηση του interface