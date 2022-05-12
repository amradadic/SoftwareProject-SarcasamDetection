import tweepy


# API keyws that yous saved earlier - consumer
api_key = 'vlY9i6EAhbLvsXkN20CaDjImR'
api_secrets = 'y1LDLRIeKKhBSeGGAQBXR82nO58EBR7TTKAEBc1km1nyqkb5BQ'
#Authentication
access_token = "1524310243236356097-GysMimoe3nlNK1ziykGwWw9dhwKY6I"
access_secret = "FKpiB0VWN1FRFyciIoXvGC6f58BxOgZJ1yw8O6muFyBUH"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(api_key, api_secrets)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

try:
    api.verify_credentials()
    print('Successful Authentication')
except:
    print('Failed authentication')


id =1086119014131208193

status = api.get_status(id)
    # fetching the text attribute
text = status.text

print("The text of the status is : \n\n" + text)









