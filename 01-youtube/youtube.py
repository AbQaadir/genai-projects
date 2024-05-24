import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from textsummary import summarize_text

def get_transcript(url):
    try:
        # Extract video ID from URL
        video_id_match = re.search(r"(?<=v=)[\w-]+", url)
        if video_id_match:
            video_id = video_id_match.group(0)
        else:
            video_id_match = re.search(r"youtu\.be/([\w-]+)", url)
            if video_id_match:
                video_id = video_id_match.group(1)
            else:
                raise ValueError("Invalid YouTube video URL")

        # Get transcripts
        transcripts = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ''
        for transcript in transcripts:
            transcript_text += transcript['text'] + ' '
        return transcript_text
    except Exception as e:
        return str(e)

def summarize_video(url):
    transcript = get_transcript(url)
    if not transcript:
        return "No transcript found for this video."
    
    
    # Split the transcript into chunks of 1024 characters
    chunk_size = 1024
    transcript_chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]

    # Summarize each chunk and concatenate the summaries
    summaries = [summarize_text(chunk) for chunk in transcript_chunks]
    summarized_text = ' '.join(summaries)
    
    return transcript, summarized_text



# steamlit app for youtube video summarization
st.title("📺YouTube Video Summarizer")
url = st.text_input("Enter the URL of a YouTube video:")
if st.button("Summarize"):
    transcript, summary = summarize_video(url)
    st.header("✍️Summary")
    st.write(summary, unsafe_allow_html=True)
    st.header("📑Transcript")
    st.write(transcript)
    
    if url:
        try:
            # Display the video
            st.header("Embedded YouTube Video:")
            st.video(url)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
st.write()
st.markdown("<p style='text-align: center;'>Copyright 2024. All rights reserved. Created by <i><b>Abdul Qaadir</b></i></p>", unsafe_allow_html=True)