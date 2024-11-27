from django.urls    import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/segment", views.segment_image, name="segment_image")
]