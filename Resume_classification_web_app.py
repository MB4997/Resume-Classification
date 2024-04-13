import PyPDF2
import docx2txt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import words
from nltk import ne_chunk, pos_tag, word_tokenize
import pickle
import pandas as pd
import re
import streamlit as st # streamlit run Resume_classification_web_app
import pickle
import numpy as np

st.title('Resume Classifier')

def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    page_no = len(reader.pages)
    for page_num in range(page_no):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text.strip()

def extract_text_from_docx(docx_path):
    return docx2txt.process(docx_path)

def cleanResume(resumeText):
    resumeText = re.sub(r'[^a-zA-Z\d]',r' ', resumeText) 
    resumeText = re.sub(r'\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText.lower()

def clean_text(text):
    text = word_tokenize(text)
    text2 = []
    for word in text:
        if word not in stopwords.words('english'):
            text2.append(WordNetLemmatizer().lemmatize(word))
    return ' '.join(text2)

def parse_resume(text):
    lines = text.split('\n')[:4]
    lines = [''.join(e for e in line if e.isalnum() or e.isspace()) for line in lines]
    lines = [line.replace('name', '') for line in lines]
    resume_text = ' '.join(lines)
    words_ = word_tokenize(resume_text)
    tagged_words = pos_tag(words_)
    nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNP']]
    english_words = set(words.words())
    names = [word for word in nouns if word.lower() not in english_words]
    return " ".join(names)

def extract_exp(text):
    Text = text.split()
    exp = []
    for words in Text:
        if words.lower() in ['year', 'yr', 'years','yrs']:
            no_index = Text.index(words) - 1
            numbers = re.findall(r'[0-9.]+', Text[no_index])
            result = ''.join(numbers)
            try:
                exp.append(float(result))
            except ValueError:
                exp.append('Not Available')
            return sum(exp)

def find_email_and_phone(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(?:\+\d{1,2}\s*)?(?:\d{3}|\(\d{3}\))[-.\s]?\d{3}[-.\s]?\d{4}\b'
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    if emails or phones:
        return emails[0] if emails else np.nan, phones[0] if phones else np.nan
    else:
        return 'Not Available','Not Available'

# Load the model and vectorization objects
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

knn_model = model_data['knn_model']
vectorizer = model_data['vectorizer']

# Add sidebar for navigation
st.sidebar.title('Navigation')
selected_page = st.sidebar.radio('Go to', ['Upload Resume', 'Shortlisted Resumes'])

# Define categories
category = {'peoplesoft': 0, 'Sql developer': 1, 'Workday': 2, 'React developer': 3}

if selected_page == 'Upload Resume':
    upload_files = st.file_uploader('Upload file',accept_multiple_files=True)

    df = { 'File Name' : [],
            'Category' : [],
            'Candidate Name' : [],
            'Years of exp' : [],
            'Email': [],
            'Contact' : []
                }

    if upload_files is not None:
        for uploaded_file in upload_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            df['File Name'].append(uploaded_file.name)

            if file_extension == 'pdf':
                resume_text = extract_text_from_pdf(uploaded_file)
                candidate_name = parse_resume(extract_text_from_pdf(uploaded_file))
                df['Candidate Name'].append(candidate_name)
                email, phone = find_email_and_phone(resume_text)
                df['Email'].append(email)
                df['Contact'].append(phone)
            elif file_extension == 'docx':
                resume_text = extract_text_from_docx(uploaded_file)
                candidate_name = parse_resume(extract_text_from_docx(uploaded_file))
                df['Candidate Name'].append(candidate_name)
                email, phone = find_email_and_phone(resume_text)
                df['Email'].append(email)
                df['Contact'].append(phone)
            else:
                st.error('Unsupported file format. Please upload a PDF or DOCX file.')

            cleaned_text = clean_text(resume_text)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = knn_model.predict(vectorized_text)[0]

            predicted_category = None
            for category_name, category_id in category.items():
                if category_id == prediction:
                    predicted_category = category_name
                    break

            if predicted_category is not None:
                category_ = predicted_category
                df['Category'].append(category_)
            else:
                category = np.nan
                st.warning("No category found for {}".format(uploaded_file.name))

            years_of_exp = extract_exp(resume_text)
            df['Years of exp'].append(years_of_exp)

        existing_data = pd.read_excel('Shortlisted_resumes.xlsx')
        appended_data = pd.concat([existing_data, pd.DataFrame(df)], ignore_index=True)
        appended_data.to_excel('Shortlisted_resumes.xlsx', index=False)

elif selected_page == 'Shortlisted Resumes':
    
    import pandas as pd
    import streamlit as st
    from io import BytesIO
    
    category = ['Sql developer','peoplesoft', 'Workday', 'React developer']
    
    selected_category = st.selectbox('Select Category', category)
    
    resume_df = pd.read_excel('Shortlisted_resumes.xlsx')
    
    # Filter the DataFrame based on the selected category
    filtered_df = resume_df[resume_df['Category'] == selected_category]
    
    # Display the filtered DataFrame in a table
    st.table(filtered_df)
    
    # Download button for the filtered data
    def download_excel():
        # Convert the filtered DataFrame to Excel format
        excel_buffer = BytesIO()
        filtered_df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        # Download the Excel file with the filtered data
        st.download_button(label="Download Excel", data=excel_buffer, file_name=f"{selected_category}_resumes.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # Call the download_excel function to display the download button
    download_excel()
    