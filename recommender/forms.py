from django import forms
# from recommender.models import cold_start

class ColdStartForm(forms.Form):
    UserID = forms.IntegerField(widget=forms.HiddenInput(), initial=1)  
    thrillers=forms.IntegerField(help_text="Interest from 1 to 10??")
    Romance=forms.IntegerField(help_text="Interest from 1 to 10??")
    Nonfiction=forms.IntegerField(help_text="Interest from 1 to 10??")
    Humor=forms.IntegerField(help_text="Interest from 1 to 10??")
    Horror=forms.IntegerField(help_text="Interest from 1 to 10??")

    # An inline class to provide additional information on the form.
    # class Meta:
    #     # Provide an association between the ModelForm and a model
    #     model = cold_start
    #     fields = ('UserID','BookTitle','BookRating')









# import ipywidgets as widgets
# from IPython.display import display

# from engine import runEngine

# button = widgets.Button(description="Click Me!")
# output = widgets.Output()

# def on_button_clicked(b):
#   # Display the message within the output widget.
#   with output:
#     runEngine(b)

# button.on_click(on_button_clicked)
# display(button, output)