from predict import predict

new_transcript = [{'start':'1', 'end':'9', 'text':'we are starting', 'video_id':'1'},
                  {'start':'10', 'end':'13', 'text':'continuing and taking a vote', 'video_id':'1'},
                 {'start':'15', 'end':'20', 'text':'on bill AB 238', 'video_id':'1'}]

x = predict(new_transcript, "NB")
print(x)