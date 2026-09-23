## PROMPT # 1:
Spot all the text and layout blocks in the image with line-level and region-level precision (such as paragraphs, headings, tables, or figures). 
Output the results strictly in JSON format as a list of dictionaries, where each entry contains the element type, its text content, and its normalized bounding box coordinates [x1, y1, x2, y2] scaled from 0 to 1000 relative to the image dimensions. 
Format: [ { "label": "text_line" | "paragraph" | "heading" | "table" | "figure", "text_content": "Extracted text string here", "bbox_2d": [x1, y1, x2, y2] } ]

## PROMPT # 2:
You are an expert document parser and optical character recognition (OCR) model. Your task is to analyze the provided image and extract all visible text with maximum accuracy. Start with analyzing document layout and then organize the extracted text into a structured, clean Markdown format. 

Follow these strict formatting rules:
1. Headers: If a section of the text appears larger, bold, or serves as a section heading, format it using Markdown headers (#, ##, ###, etc.).
2. Lists: Format all bullet points, checkboxes, or numbered items using proper Markdown syntax (- or 1.).
3. Tables: Reconstruct tables completely using standard Markdown table format (| Column | Column |). Ensure columns align properly based on the image layout. Handle Missing Data: If a cell or specific property is missing for a given row, leave it blank (` | | `) or populate it with "N/A" only if contextually appropriate. Do not shift cells left or right to fill gaps. Header Alignment: Use appropriate Markdown colon syntax (`:---` for left-aligned, `:---:` for centered, `---:` for right-aligned) in the delimiter row based on data type (e.g., text left, numbers right).
4. Paragraphs: Maintain separate text blocks as clear paragraphs. 
5. Text Styling: Apply _italics_ or **bold** text styles where they are visually emphasized in the original image.
6. Links & Metadata: Format visible URLs or hyperlinks using [text](url).
7. Math and Code: Render mathematical equations in LaTeX (using $ or $$) and source code in proper triple-backtick markdown blocks.

Output Constraints:
- Do not include conversational preambles, introductory filler ("Here is the text..."), or any post-text explanations. 
- Provide ONLY the pure Markdown transcription.
- If the image contains visual-only elements (like graphs or charts), reconstruct them completely using standard Markdown table format.
