Check the Live Demo Using Gradio Interface - https://customer-support-ticket-classifier-using-svm-ml-production.up.railway.app/

give this command for shortcut and check app - Can you tell me more about the UltraClean Vacuum warranty? Also, is it available in white? 

Problem Statement
Objective:
Develop a machine learning pipeline that classifies customer support tickets by their issue
type and urgency level, and extracts key entities (e.g., product names, dates, complaint
keywords). The file (ai_dev_assignment_tickets_complex_1000 ) is provided.

1. Data Preparation
• You are provided with an Excel file (ai_dev_assignment_tickets_complex_1000 )
containing anonymized customer support ticket data with the following columns:
o ticket_id
o ticket_text
o issue_type (label)
o urgency_level (label: Low, Medium, High)
o product (ground truth for entity extraction)
• Clean and preprocess the data, including but not limited to:
o Text normalization (lowercase, removing special characters)
o Tokenization, stopword removal, lemmatization
o Handling missing data
2. Feature Engineering
• Create meaningful features using traditional NLP techniques:
o Bag-of-Words, TF-IDF, or similar
o Extract additional features such as ticket length, sentiment score, etc.
• Justify your feature selection in the documentation.
3. Multi-Task Learning
• Build and train two separate models:
1. Issue Type Classifier: Predict the ticket's issue_type (multi-class classification)
2. Urgency Level Classifier: Predict the urgency_level (multi-class classification)
• Use any classical ML models (e.g., Logistic Regression, SVM, Random Forest, etc.).

4. Entity Extraction
5. • Using traditional NLP, extract key entities from ticket_text:
o Product names (from a provided product list or using regex/rule-based
methods)
o Dates mentioned in the text
o Complaint keywords (e.g., “broken”, “late”, “error”)
• Return extracted entities as a dictionary.
5. Integration
• Write a function that takes raw ticket_text as input and returns:
o Predicted issue_type
o Predicted urgency_level
o Extracted entities (as a JSON/dictionary)
6. Gradio Interface (Optional)
• Build an interactive Gradio web app where users can:
o Input raw ticket text
o See the predicted issue type, urgency, and extracted entities as output
