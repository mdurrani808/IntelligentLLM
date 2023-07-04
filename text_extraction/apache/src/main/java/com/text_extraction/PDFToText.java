package com.text_extraction;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

public class PDFToText {
    public static void main(String[] args) throws IOException {
        // Set the directory containing the PDF files
        File pdfDirectory = new File("files");
        // Create a PDFTextStripper object to extract text from the PDFs
        PDFTextStripper pdfStripper = new PDFTextStripper();
        // Iterate over the files in the directory
        try {
            for (File pdfFile : pdfDirectory.listFiles()) {
                // Load the PDF document
                PDDocument document = PDDocument.load(pdfFile);
                // Extract the text from the document
                String text = pdfStripper.getText(document);
                // Close the document
                document.close();
                // Write the extracted text to a file with the same name as the PDF, but with a .txt extension
                String txtFileName = pdfFile.getName().replace(".pdf", ".txt");
                Files.write(Paths.get(txtFileName), text.getBytes());
            }
        } catch(NullPointerException e) {
            System.out.println("No files found!");
        }
        
    }
}