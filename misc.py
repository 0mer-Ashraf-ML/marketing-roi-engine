# Generate sample data programmatically
from data.sample_data import SampleDataGenerator

generator = SampleDataGenerator(seed=42)
sample_data = generator.generate_all_sample_data(days=90)

# This creates:
print(f"Campaigns: {len(sample_data['campaigns']):,} records")
print(f"Keywords: {len(sample_data['keywords']):,} records") 
print(f"Products: {len(sample_data['products']):,} records")
print(f"Financial: {len(sample_data['financial']):,} records")
print(f"Attribution: {len(sample_data['attribution']):,} records")
print(f"Competitive: {len(sample_data['competitive']):,} records")

# Save to CSV files
from data.sample_data import save_sample_data_to_csv
save_sample_data_to_csv(sample_data, "sample_data/")