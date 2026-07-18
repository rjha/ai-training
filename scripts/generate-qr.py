import qrcode

# Define the data you want to encode
data = "https://forms.gle/taPKwXCXxcxyUiuF9"

# Create and save the QR code image
img = qrcode.make(data)
img.save("tech_talkied_qrcode.png")