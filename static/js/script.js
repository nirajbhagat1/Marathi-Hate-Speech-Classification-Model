document.addEventListener('DOMContentLoaded', () => {
    const textForm = document.querySelector('form[action="/classify-text"]');
    const csvForm = document.querySelector('form[action="/classify-csv"]');

    // Validate text input form
    textForm.addEventListener('submit', (e) => {
        const textInput = textForm.querySelector('textarea[name="text"]');
        if (textInput.value.trim() === '') {
            alert('Please enter some text to classify.');
            e.preventDefault(); // Prevent form submission
        }
    });

    // Validate CSV upload form
    csvForm.addEventListener('submit', (e) => {
        const fileInput = csvForm.querySelector('input[type="file"]');
        if (fileInput.files.length === 0) {
            alert('Please upload a CSV file.');
            e.preventDefault(); // Prevent form submission
        }
    });
});
