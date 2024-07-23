
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import json
from .model import predict_spam

def index(request):
    return render(request,'index.html')


@csrf_exempt
def predict_spam_view(request):
    if request.method == "POST":
        message = request.POST.get("input_text", "")
        if not message:
            return JsonResponse({"error": "No message provided"}, status=400)
        
        prediction = predict_spam(message)
        return JsonResponse({"result": prediction})
    return render(request, 'index.html')

