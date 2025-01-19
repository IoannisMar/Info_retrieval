import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import string
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.probability import FreqDist
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

stop_words = set(stopwords.words('english'))#Δημιουργία συνόλου με τις stopwords της αγγλικής γλώσσας
punctuation_table = str.maketrans('', '', string.punctuation)#Δημιουργία πίνακα για αφαίρεση σημείων στίξης

def parse_all_file(file_path):
 data = []#Λίστα για αποθήκευση των εγγραφών
 current_record = {}#Δομή για να αποθηκεύουμε κάθε εγγραφή με τα πεδία της
 with open(file_path, 'r', encoding='utf-8') as f:#Άνοιγμα του αρχείου για ανάγνωση
     for line in f:#Για κάθε γραμμή του αρχείου
        line = line.strip()#Αφαίρεση κενών χαρακτήρων από την αρχή και το τέλος της γραμμής
        if line.startswith('.I'):#Αν η γραμμή ξεκινά με .I (ID εγγράφου)
            if current_record:  #Αν υπάρχει ήδη ένα εγγεγραμμένο έγγραφο, το αποθηκεύουμε
                    data.append(current_record)
            current_record = {"ID": int(line[2:].strip())}#Δημιουργία νέας εγγραφής με το ID
        elif line.startswith('.T'):#Αν η γραμμή ξεκινά με .T(Τίτλος)
                current_record["Title"] = ""#Αρχικοποίηση του τίτλου
        elif line.startswith('.A'):#Αν η γραμμή ξεκινά με .A(Συγγραφέας)
                current_record["Author"] = ""#Αρχικοποίηση του συγγραφέα
        elif line.startswith('.W'):#Αν η γραμμή ξεκινά με .W(Περίληψη)
                current_record["Abstract"] = ""#Αρχικοποίηση της περίληψης
        elif line.startswith('.X'):#Αν η γραμμή ξεκινά με .(Διασταυρώσεις)
                current_record["Cross_References"] = ""#Αρχικοποίηση των διασταυρώσεων
        else:
         #Προσθήκη κειμένου στα αντίστοιχα πεδία
         if "Title" in current_record and not current_record["Title"]:
             current_record["Title"] += line
         elif "Author" in current_record and not current_record["Author"]:
             current_record["Author"] += line
         elif "Abstract" in current_record and not current_record["Abstract"]:
             current_record["Abstract"] += line
         elif "Cross_References" in current_record and not current_record["Cross_References"]:
             current_record["Cross_References"] += line
 if current_record:#Αποθήκευση του τελευταίου εγγράφου
   data.append(current_record)
 return pd.DataFrame(data)#Επιστροφή του DataFrame με τα δεδομένα
file_path = 'CISI.ALL'#Ορισμός του path του αρχείου
df = parse_all_file(file_path)#Ανάγνωση του αρχείου και αποθήκευση των δεδομένων σε DataFrame

def clean_and_tokenize(text):
 tokens = word_tokenize(text.lower())#Μετατροπή των γραμμάτων σε πεζά και ταυτόχρονο tokenization
 cleaned_tokens = [token.translate(punctuation_table) for token in tokens if token not in stop_words and token.isalnum()]#Αφαίρεση των stop words και των σημείων στίξης
 return cleaned_tokens#Επιστροφή των καθαρών tokens

df['Abstract'] = df['Abstract'].apply(lambda x: clean_and_tokenize(x))#Καθαρισμός της στήλης 'Abstract'
df['Title'] = df['Title'].apply(lambda x: clean_and_tokenize(x))#Καθαρισμός της στήλης 'Title'
print(df)#Εκτύπωση του DataFrame για έλεγχο
all_tokens = [token for tokens in df['Abstract'] for token in tokens]#Συναρμολόγηση όλων των tokens
fdist1 = FreqDist(all_tokens)#Δημιουργία συχνότητας εμφάνισης λέξεων
fdist1.plot(50, title='Top 50 Most Common Tokens')#Εμφάνιση γραφικής παράστασης με τις 50 πιο κοινές λέξεις
inverted_index = defaultdict(set)#Δημιουργία ενός ανεστραμμένου ευρετηρίου

for _, row in df.iterrows():#Για κάθε γραμμή του DataFrame
    doc_id = row['ID']#Λήψη του ID του εγγράφου
    for token in row['Abstract'] + row['Title']:#Για κάθε token στην περίληψη και στον τίτλο
        inverted_index[token].add(doc_id)#Προσθήκη του doc_id στο αντίστοιχο token του ευρετηρίου
inverted_index_csv = [{"Term": term, "Document IDs": ",".join(map(str, sorted(doc_ids)))} for term, doc_ids in inverted_index.items()]
inverted_index_df = pd.DataFrame(inverted_index_csv)#Δημιουργία DataFrame από το ανεστραμμένο ευρετήριο
inverted_index_df.to_csv('inverted_index.csv', index=False, encoding='utf-8')#Αποθήκευση του DataFrame σε αρχείο CSV

df.to_csv('ApotelesmataXarakthrwn.csv', index=False, encoding='utf-8')#αποθήκευση του καθαρισμένου DataFrame σε αρχείο CSV

inverted_index = pd.read_csv('inverted_index.csv')#Φόρτωση του ανεστραμμένου ευρετηρίου
df = pd.read_csv('ApotelesmataXarakthrwn.csv')#Φόρτωση του καθαρισμένου DataFrame
corpus = df['Abstract'].apply(lambda x: ' '.join(eval(x))).tolist()#Δημιουργία της λίστας του corpus (ενοποιημένα strings από tokens)
vectorizer = CountVectorizer(stop_words=None, min_df=1)#Δημιουργία του vectorizer για το TF-IDF
X = vectorizer.fit_transform(corpus)#Εφαρμογή του vectorizer στο corpus

def search_query(query, algorithm):#Συνάρτηση search_query για την υλοποίηση των αναζητήσεων
 query_tokens = word_tokenize(query.lower())#Query_tokens περιλαμβάνει τα tokens του query που έδωσε ο χρήστης
 if algorithm == "Boolean Retrieval":#ΠΕΡΙΠΤΩΣΗ 1: Boolean Retrieval
  if "and" in query_tokens:#ΠΕΡΙΠΤΩΣΗ 1.1: Επιλογή AND
   terms = query.split('AND')
   terms = [term.strip() for term in terms]
            
   doc_sets = []#Ανακτάμε τα έγγραφα για την συγκρίσεις και την εύρεση κοινών
   found_terms = set()  #Δημιουργία συνόλου για να καταγράφουμε τους όρους που βρέθηκαν

   for term in terms:#Για τον αριθμό των terms υλοποιούμε επανάληψη for
    result = inverted_index[inverted_index['Term'].isin(word_tokenize(term.lower()))]#Θέτουμε την εντολή result
    if not result.empty:#Σε περίπτωση που το result δεν είναι κενό
     document_ids = set()#Δημιουργία κενού συνόλου στο document_ids
     for term in word_tokenize(term.lower()):#Για term διαχωρίζει την λέξη και διασφαλίζει την περίπτωση case-insensitive
      docs = result[result['Term'] == term]['Document IDs'].values#Πραγματοποίηση της αναζήτησης στο ευρετήριο και επιστροφή των IDs των αρχείων που βρέθηκε
      if docs.size > 0:#Έλεγχος αν βρέθηκε σε τουλάχιστον 1 έγγραφο
       doc_ids = set(map(int, docs[0].split(',')))#Δημιουργία map τύπου int για τα doc_ids
       document_ids.update(doc_ids)#Προσθήκη IDs στο document_ids
     if document_ids:  # Αν βρέθηκαν έγγραφα για τον όρο
      found_terms.add(term)  # Προσθήκη του όρου στα ευρεθέντα
      doc_sets.append(document_ids) #Προσθήκη του συνόλου των εγγράφων document_ids στο doc_sets
     
   if len(found_terms) < len(terms):  #Αν κάποιο term δεν βρέθηκε
    missing_terms = set(terms) - found_terms  #Βρίσκουμε ποιοι όροι δεν βρέθηκαν
    return [f"Δεν βρέθηκαν αρχεία που να περιέχουν και τις δύο λέξεις. "
            f"Λείπουν οι όροι: {', '.join(missing_terms)}."]

   common_docs = set.intersection(*doc_sets) if doc_sets else set()  #Αν το doc_sets είναι κενό, επιστρέφουμε κενό σύνολο

   if not common_docs:  #Αν δεν υπάρχουν κοινά έγγραφα
    return [f"Δεν βρέθηκαν αρχεία που να περιέχουν και τις δύο λέξεις."]

   results = [f"{df[df['ID'] == doc_id]['ID'].values[0]}" for doc_id in common_docs]
   return results#επιστροφή αποτελεσμάτων

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

   results = [f"{df[df['ID'] == doc_id]['ID'].values[0]}" for doc_id in document_ids]#Εμφάνιση μηνύματος και των εγγράφων που βρέθηκαν οι λέξεις
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
   results = [f"{df[df['ID'] == doc_id]['ID'].values[0]}" for doc_id in final_results]#Εμφάνιση μηνύματος και των εγγράφων που βρέθηκαν οι λέξεις
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
            
   results = [f"{df[df['ID'] == doc_id]['ID'].values[0]}" for doc_id in document_ids]#Εμφάνιση μηνύματος και των εγγράφων που βρέθηκαν οι λέξεις
   return results if results else [f"Δεν βρέθηκαν αποτελέσματα για το '{query}'."]#Εμφάνιση σε περίπτωση που δεν υπήρχαν αρχεία

########################################################################################################################################3

 elif algorithm == "TF-IDF":
    query_vector = vectorizer.transform([query]) #Μετασχηματίζει το ερώτημα σε διανύσματα TF-IDF
    scores = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()#Υπολογίζει τα σκορ (similarity) μεταξύ του ερωτήματος και των εγγράφων
    ranked_indices = np.argsort(scores)[::-1]#Ταξινομεί τα σκορ κατά φθίνουσα σειρά
    results = [f"{df.iloc[i]['Selida']} " for i in ranked_indices if scores[i] > 0]#Δημιουργεί τα αποτελέσματα (ταξινομημένα) με βάση τα σκορ TF-IDF
    return results if results else [f"Δεν βρέθηκαν αποτελέσματα για '{query}' με TF-IDF."]#Επιστρέφει τα αποτελέσματα ή μήνυμα αν δεν βρέθηκαν

########################################################################################################################################

 elif algorithm == "Vector Space Model":
    query_vector = vectorizer.transform([query])#Μετασχηματίζει το ερώτημα σε διανύσματα TF-IDF  
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()#Υπολογίζει την ομοιότητα (cosine similarity) μεταξύ του ερωτήματος και των εγγράφων  
    ranked_indices = np.argsort(similarities)[::-1]#Ταξινομεί τις ομοιότητες κατά αύξουσα σειρά (χωρίς το[::-1])
    results = [f"{df.iloc[i]['Selida']} | Ομοιότητα: {similarities[i]:.4f}" for i in ranked_indices if similarities[i] > 0]#Δημιουργεί τα αποτελέσματα (ταξινομημένα) με βάση την ομοιότητα
    return results if results else [f"Δεν βρέθηκαν αποτελέσματα για '{query}' με το Vector Space Model."]#Επιστρέφει τα αποτελέσματα ή μήνυμα αν δεν βρέθηκαν


def read_queries_from_file(file_path):
    queries = {}#Δημιουργία λεξικού για αποθήκευση των ερωτημάτων
    with open(file_path, 'r') as file:#Άνοιγμα του αρχείου για ανάγνωση
     lines = file.readlines()#Ανάγνωση όλων των γραμμών του αρχείου
     query_id = None#Αρχικοποίηση του query_id (θα περιέχει το ID της ερώτησης)
     query_text = []#Λίστα για αποθήκευση του κειμένου του κάθε ερωτήματος

     for line in lines:#Διαδοχική επεξεργασία κάθε γραμμής του αρχείου
         line = line.strip()#Αφαίρεση των λευκών χαρακτήρων στην αρχή και στο τέλος της γραμμής
         if line.startswith(".I"):#Εύρεση του ID της ερώτησης
           if query_id is not None:#Αν έχει ήδη συλλεχθεί προηγούμενη ερώτηση
              queries[query_id] = " ".join(query_text)#Επιστρέφουμε το query ως string
           query_id = line.split()[1]#Αποθήκευση του ID
           query_text = []#Εκκίνηση νέας λίστας για την τρέχουσα ερώτηση
         elif line.startswith(".W"):#Εύρεση της αρχής του query text
            continue
         else:
          query_text.append(line)#Προσθήκη κειμένου στην ερώτηση

         if query_id is not None:#Αποθήκευση της τελευταίας ερώτησης
            queries[query_id] = " ".join(query_text)#Αποθήκευση του τελευταίου ερωτήματος στο λεξικό
     return queries #Επιστροφή του λεξικού με όλα τα ερωτήματα και τα κείμενά τους



def search_query_with_evaluation(algorithm, relevant_docs, output_file_path):
   results_to_write = []#Δημιουργία μιας λίστας για την αποθήκευση των αποτελεσμάτων
   for query_id, query_text in queries.items():#Για κάθε ερώτημα στο λεξικό των queries
    results = search_query(query_text, algorithm)#Αναζητούμε τα αποτελέσματα για το query χρησιμοποιώντας τον αλγόριθμο
      
    if results:#Αν υπάρχουν αποτελέσματα για το query
      for doc_id in results:#Για κάθε έγγραφο στα αποτελέσματα
        results_to_write.append((query_id, doc_id))#Προσθέτουμε το query_id και το doc_id στη λίστα των αποτελεσμάτων
   results_to_write.sort(key=lambda x: (int(x[0]), int(x[1])))#Ταξινόμηση των αποτελεσμάτων πρώτα με το query_id και μετά με το doc_id
   with open(output_file_path, 'w') as file:#Άνοιγμα του αρχείου για εγγραφή των αποτελεσμάτων
    for query_id, doc_id in results_to_write:#Για κάθε ζεύγος query_id και doc_id
      file.write(f"{query_id} {doc_id}\n")#Εγγραφή του query_id και του doc_id στο αρχείο
       
queries = read_queries_from_file("CISI.QRY")#Καλούμε τη συνάρτηση για να διαβάσουμε τα queries από το αρχείο

def read_relevant_docs(rel_file_path):
   relevant_docs = {}#Δημιουργία λεξικού για να αποθηκεύσουμε τα σχετικά έγγραφα
   with open(rel_file_path, 'r') as file:#Άνοιγμα του αρχείου με τα σχετικά έγγραφα
     for line in file:#Για κάθε γραμμή στο αρχείο
       parts = line.split()#Διαχωρίζουμε τη γραμμή σε μέρη
       query_id = parts[0]#Πρώτο στοιχείο: ID της ερώτησης
       doc_id = parts[1]#Δεύτερο στοιχείο: ID του εγγράφου
       if query_id not in relevant_docs:#Αν το query_id δεν υπάρχει ήδη στο λεξικό
        relevant_docs[query_id] = set()#Δημιουργούμε ένα νέο σύνολο για το query_id
       relevant_docs[query_id].add(doc_id)#Προσθέτουμε το doc_id στο σύνολο του query_id
   return relevant_docs#Επιστρέφουμε το λεξικό με τα σχετικά έγγραφα

rel_file_path = 'CISI.REL'#Ορίζουμε το path του αρχείου με τα relevant έγγραφα
relevant_docs = read_relevant_docs(rel_file_path)#Διαβάζουμε τα relevant έγγραφα

output_file_path = 'search_results.REL'#Ορίζουμε το όνομα του αρχείου εξόδου
search_query_with_evaluation("Boolean Retrieval", relevant_docs, output_file_path)#Εκτελούμε τη συνάρτηση για αναζήτηση και αξιολόγηση
print("Το αρχείο δημιουργήθηκε.")