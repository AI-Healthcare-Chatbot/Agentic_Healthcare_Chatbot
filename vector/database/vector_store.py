import os
import pandas as pd
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MetadataVectorizer:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_metadata_index(self, index_name='healthcare-metadata-13-april'):
        spec = ServerlessSpec(cloud='aws', region='us-east-1')
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=spec
            )
            import time
            while not self.pc.describe_index(index_name).status['ready']:
                print("⏳ Waiting for index to be ready...")
                time.sleep(1)

        return self.pc.Index(name=index_name)

    def prepare_metadata_embeddings(self, metadata_df):
        """
        Prepare embeddings for metadata using combined fields
        """
        metadata_df['combined_text'] = metadata_df.apply(
            lambda row: f"Column: {row['column_name']}. "
                f"Description: {row['description']}. "
                f"Data Type: {row['data_type']}. "
                f"Example Values: {row['example_values']}. "
                f"Related Concepts: {row['related_concepts']}. "
                f"User Questions: {row['possible_user_questions']}.",
            axis=1
        )
        
        embeddings = self.embedding_model.encode(metadata_df['combined_text'].tolist())
        return metadata_df, embeddings

    def upload_to_pinecone(self, index, metadata_df, embeddings):
        """
        Upload metadata vectors to Pinecone
        """
        vectors = []
        for i, (_, row) in enumerate(metadata_df.iterrows()):
            vector = {
                'id': f'metadata_{row["column_name"]}',
                'values': embeddings[i].tolist(),
                'metadata': {
                    'column_name': row['column_name'],
                    'description': row['description'],
                    'data_type': row['data_type'],
                    'example_values': row['example_values'],
                    'related_concepts': row['related_concepts'],
                    'possible_user_questions': row['possible_user_questions'],
                    'combined_text': row['combined_text']  # crucial for semantic search
                }
            }
            vectors.append(vector)

        index.upsert(vectors)
        print(f"✅ Uploaded {len(vectors)} metadata vectors to Pinecone")

    def query_metadata(self, index, query_text, top_k=5):
        query_embedding = self.embedding_model.encode([query_text])[0].tolist()
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results

def display_query_results(results):
    if not results['matches']:
        print("No matches found.")
        return

    print("\nTop Matches:")
    for i, match in enumerate(results['matches'], start=1):
        md = match['metadata']
        print(f"{i}. Column: {md['column_name']}")
        print(f"   Description: {md['description']}")
        print(f"   Data Type: {md['data_type']}")
        print(f"   Example Values: {md['example_values']}")
        print(f"   Related Concepts: {md['related_concepts']}")
        print(f"   Possible User Questions: {md['possible_user_questions']}")
        print(f"   Similarity Score: {match['score']:.4f}\n")
def main():
    metadata_file_path = os.getenv('METADATA_FILE_PATH', 'data/metadata/healthcare_metadata_13th_April.xlsx')
    metadata_df = pd.read_excel(os.path.abspath(metadata_file_path))

    vectorizer = MetadataVectorizer()
    index = vectorizer.create_metadata_index()
    prepared_df, embeddings = vectorizer.prepare_metadata_embeddings(metadata_df)
    vectorizer.upload_to_pinecone(index, prepared_df, embeddings)

    print("📊 Index Stats:")
    print(index.describe_index_stats())

    test_query = "what is the cost for male patients who want to get heart surgery done?"
    print(f"\n🔍 Running query: {test_query}")
    results = vectorizer.query_metadata(index, test_query)
    display_query_results(results)

if __name__ == "__main__":
    main()
