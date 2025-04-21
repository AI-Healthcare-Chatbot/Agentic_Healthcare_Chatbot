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
        # Initialize Pinecone with new method
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_metadata_index(self, index_name='healthcare-metadata'):
        """
        Create Pinecone index for metadata
        """
        # Specify the spec
        spec = ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
        
        existing_indexes = [index.name for index in self.pc.list_indexes()]

        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=spec
            )

            # Wait for readiness
            import time
            while not self.pc.describe_index(index_name).status['ready']:
                print("⏳ Waiting for index to be ready...")
                time.sleep(1)

        return self.pc.Index(index_name)

    # ... rest of the code remains the same
    
    def prepare_metadata_embeddings(self, metadata_df):
        """
        Prepare embeddings for metadata
        """
        # Combine metadata columns
        metadata_df['combined_text'] = metadata_df.apply(
            lambda row: f"Column: {row['Column Name']}. " +
                        f"Description: {row['Description']}. " +
                        f"API Field: {row['API Field Name']}", 
            axis=1
        )
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            metadata_df['combined_text'].tolist()
        )
        
        return metadata_df, embeddings
    
    def upload_to_pinecone(self, index, metadata_df, embeddings):
        """
        Upload metadata vectors to Pinecone
        """
        # Prepare vectors for upload
        vectors = []
        for i, (_, row) in enumerate(metadata_df.iterrows()):
            vector = {
                'id': f'metadata_{row["Column Name"]}',
                'values': embeddings[i].tolist(),
                'metadata': {
                    'column_name': row['Column Name'],
                    'description': row['Description'],
                    'api_field_name': row['API Field Name']
                }
            }
            vectors.append(vector)
        
        # Upsert vectors
        index.upsert(vectors)
        
        print(f"Uploaded {len(vectors)} metadata vectors to Pinecone")
    
    def query_metadata(self, index, query_text, top_k=5):
        """
        Perform semantic search on metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0].tolist()
        
        # Perform similarity search
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results

def main():
    # Load metadata
    metadata_file_path = os.getenv('METADATA_FILE_PATH', 'Data/metadata/healthcare_metadata.xlsx')
    metadata_df = pd.read_excel(os.path.abspath(metadata_file_path))

    # Initialize vectorizer
    vectorizer = MetadataVectorizer()

    # Create index
    index = vectorizer.create_metadata_index()

    # Prepare embeddings
    prepared_df, embeddings = vectorizer.prepare_metadata_embeddings(metadata_df)

    # Upload to Pinecone
    vectorizer.upload_to_pinecone(index, prepared_df, embeddings)

    # Print index stats
    print("Index Stats:")
    print(index.describe_index_stats())

    # DEBUG: Print a sample embedded vector
    print("\nSample vector embedding length:", len(embeddings[0]))
    print("Sample combined text:\n", prepared_df['combined_text'].iloc[0])

    # Define your test query
    test_query = "Total estimated cost for the discharge"

    # DEBUG: Print query vector length
    query_embedding = vectorizer.embedding_model.encode([test_query])[0].tolist()
    print("\nQuery vector length:", len(query_embedding))

    # Run query
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    # DEBUG: Print raw result
    print("\nRAW QUERY RESULT:")
    print(results)

    # Show final matches
    print("\nQuery Results:")
    display_query_results(results)
    
def display_query_results(results):
    """
    Helper function to display query results
    """
    if not results['matches']:
        print("No matches found.")
        return

    print("\nTop Matches:")
    for i, match in enumerate(results['matches'], start=1):
        print(f"{i}. Column: {match['metadata']['column_name']}")
        print(f"   Description: {match['metadata']['description']}")
        print(f"   API Field: {match['metadata']['api_field_name']}")
        print(f"   Similarity Score: {match['score']:.4f}\n")


if __name__ == "__main__":
    main()
