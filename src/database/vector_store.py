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
            cloud='gcp',
            region='us-central1'
        )
        
        # Check if index exists and delete if necessary
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if index_name in existing_indexes:
            self.pc.delete_index(index_name)
        
        # Create new index
        self.pc.create_index(
            name=index_name,
            dimension=384,  # Embedding dimension
            metric='cosine',
            spec=spec  # Add the spec argument
        )
        
        # Get the index
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
    
    # Test query
    results = vectorizer.query_metadata(index, test_query)
    print("\nQuery Results:")
    display_query_results(results)
def display_query_results(results):
    """
    Helper function to display query results
    """
    print("\nQuery Results:")
    for match in results['matches']:
        print(f"Column: {match['metadata']['column_name']}")
        print(f"Description: {match['metadata']['description']}")
        print(f"Similarity Score: {match['score']}\n")

if __name__ == "__main__":
    main()
