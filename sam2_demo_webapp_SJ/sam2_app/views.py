from django.shortcuts import render
from DoSamMask import make_masked_img
from django.http import HttpResponse

# Create your views here.
def index(request):
    return render(request, "index.html")

def segment_image(request):
    if request.method == "POST":
        image = list(request.FILES.values())[0]
        json_points = request.POST.get('points')

        masked_image = make_masked_img(image,json_points)

        response = HttpResponse(content_type="image/png")
        masked_image.save(response, "PNG")
        return response