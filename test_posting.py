from model import FakeNewsModel

def test_content_posting():
    # Initialize the model with a small number of agents for testing
    model = FakeNewsModel(N=10, m_links=2, news_amount=50, fake_news_percentage=20, 
                          bot_percentage=20, influencer_percentage=20, seed=42)

    # Run the model for a few steps
    for _ in range(5):
        model.step()

    # Print out the number of content pieces and some details
    print(f"Total content pieces: {len(model.news_content)}")
    fake_content_count = sum(1 for content in model.news_content if content.isFake)
    print(f"Fake content pieces: {fake_content_count}")
    print(f"Real content pieces: {len(model.news_content) - fake_content_count}")

    # Print details of the first few content pieces
    for i, content in enumerate(model.news_content[:5]):
        print(f"Content {i}: Fake={content.isFake}, Topic Vector={content.topic_vector}")

if __name__ == "__main__":
    test_content_posting() 