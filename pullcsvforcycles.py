import requests
import argparse
import sys
import subprocess

def download_csv(url, output_filename):
    """
    Download a CSV file from the provided URL and save it locally with the specified filename.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError if the response was an error
    except requests.RequestException as e:
        print("Error downloading CSV:", e)
        sys.exit(1)
    
    with open(output_filename, 'wb') as f:
        f.write(response.content)
    print(f"CSV downloaded successfully and saved as '{output_filename}'.")

def run_analysis(script_name, csv_filename):
    """
    Run the specified analysis script, passing the CSV filename as an argument.
    """
    try:
        subprocess.run(["python", script_name, csv_filename], check=True)
    except subprocess.CalledProcessError as e:
        print("Error running analysis script:", e)
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download a CSV file from a URL, save it with a given name, and optionally run an analysis script."
    )
    parser.add_argument('--url', required=True, help="URL of the CSV file to download")
    parser.add_argument('--output', required=True, help="Output filename to save the CSV")
    parser.add_argument('--analysis', help="(Optional) Path to the analysis script to run after downloading")
    
    args = parser.parse_args()
    
    # Download the CSV file
    download_csv(args.url, args.output)
    
    # Optionally run the analysis script if provided
    if args.analysis:
        print(f"Running analysis script '{args.analysis}' with CSV file '{args.output}'...")
        run_analysis(args.analysis, args.output)
