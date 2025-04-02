import openai

# Load your OpenAI API key
openai.api_key = 'your-api-key'

def generate_descriptions(objects):
    descriptions = []
    for obj in objects:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Describe a {obj} in detail.",
            max_tokens=50
        )
        description = response.choices[0].text.strip()
        descriptions.append(description)
    return descriptions

if __name__ == "__main__":
    objects = ['cat', 'dog', 'car']  # Example list of detected objects
    descriptions = generate_descriptions(objects)
    print("Generated descriptions:")
    for obj, desc in zip(objects, descriptions):
        print(f"{obj}: {desc}")
