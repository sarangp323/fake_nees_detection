from django.shortcuts import render
from rest_framework.decorators import api_view
import pickle

classifier = pickle.load(open('fakenews/randomforest.pkl', 'rb'))
cv = pickle.load(open('fakenews/countvector.pkl', 'rb'))
tfidf = pickle.load(open('fakenews/tfidftransformer.pkl', 'rb'))

# Create your views here.
@api_view(['GET',])
def home(request):
    return render(request, 'home.html')

@api_view(['POST',])
def predict(request):

    news = request.data['news']
    
    data = cv.transform([news]).toarray()
    data = tfidf.transform(data)
    
    prediction = classifier.predict(data)
    data = {
        "prediction": prediction[0]
    }
    return render(request, 'predict.html', data)
