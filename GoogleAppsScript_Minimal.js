/**
 * Minimal Working Google Apps Script - No Complex Operations
 * This version avoids operations that might cause "not supported" errors
 */

function createMinimalWorkingForm() {
  try {
    Logger.log("Creating minimal working form...");

    // Create form
    var form = FormApp.create('AI Safety Testing Configuration');
    Logger.log("Form created successfully with ID: " + form.getId());

    // Set basic description
    form.setDescription('This form helps us customize AI safety testing for your industry. Time required: 15-20 minutes.');

    // Basic settings
    form.setCollectEmail(false);
    form.setLimitOneResponsePerUser(false); // Try without limit first

    Logger.log("Adding questions...");

    // Question 1: Industry
    var q1 = form.addMultipleChoiceItem();
    q1.setTitle('What industry do you work in?');
    q1.setChoices([
      q1.createChoice('Healthcare/Medical'),
      q1.createChoice('Financial Services'),
      q1.createChoice('Legal Services'),
      q1.createChoice('Technology'),
      q1.createChoice('Government'),
      q1.createChoice('Education'),
      q1.createChoice('Other')
    ]);
    q1.setRequired(true);
    Logger.log("Question 1 added");

    // Question 2: Specialization
    var q2 = form.addTextItem();
    q2.setTitle('Your area of expertise within this industry');
    q2.setHelpText('e.g., Cardiology, Investment Banking, Corporate Law');
    Logger.log("Question 2 added");

    // Question 3: AI Use Cases
    var q3 = form.addParagraphTextItem();
    q3.setTitle('How does your organization use AI systems?');
    q3.setHelpText('Describe current or planned AI applications');
    q3.setRequired(true);
    Logger.log("Question 3 added");

    // Question 4: Failure Scenarios
    var q4 = form.addParagraphTextItem();
    q4.setTitle('What are the most dangerous AI failure scenarios in your industry?');
    q4.setHelpText('Describe worst-case scenarios where AI could cause harm');
    q4.setRequired(true);
    Logger.log("Question 4 added");

    // Question 5: Regulations
    var q5 = form.addParagraphTextItem();
    q5.setTitle('What regulations must AI systems comply with in your industry?');
    q5.setHelpText('List relevant laws, regulations, or professional standards');
    q5.setRequired(true);
    Logger.log("Question 5 added");

    // Question 6: Unauthorized Practice
    var q6 = form.addParagraphTextItem();
    q6.setTitle('What should AI systems NEVER do in your industry?');
    q6.setHelpText('What would constitute unauthorized professional practice?');
    q6.setRequired(true);
    Logger.log("Question 6 added");

    // Question 7: Professional Terms
    var q7 = form.addParagraphTextItem();
    q7.setTitle('Key technical terms and jargon in your industry');
    q7.setHelpText('List important professional vocabulary and technical terms');
    q7.setRequired(true);
    Logger.log("Question 7 added");

    // Question 8: Authority Claims
    var q8 = form.addParagraphTextItem();
    q8.setTitle('Words/phrases used to claim authority in your industry');
    q8.setHelpText('Language that scammers might use to sound legitimate');
    q8.setRequired(true);
    Logger.log("Question 8 added");

    // Question 9: Red Flags
    var q9 = form.addParagraphTextItem();
    q9.setTitle('Language that should raise immediate concern');
    q9.setHelpText('Words/phrases that indicate fraud or dangerous advice');
    q9.setRequired(true);
    Logger.log("Question 9 added");

    // Question 10: Safety Responses
    var q10 = form.addParagraphTextItem();
    q10.setTitle('How should AI respond to inappropriate requests?');
    q10.setHelpText('What should AI say when asked for unauthorized professional advice?');
    q10.setRequired(true);
    Logger.log("Question 10 added");

    // Question 11: Compliance Level
    var q11 = form.addMultipleChoiceItem();
    q11.setTitle('How strict should safety measures be?');
    q11.setChoices([
      q11.createChoice('Extremely strict - Zero tolerance'),
      q11.createChoice('Very strict - Educational info only'),
      q11.createChoice('Moderately strict - General info only'),
      q11.createChoice('Flexible - Broad guidance with disclaimers')
    ]);
    q11.setRequired(true);
    Logger.log("Question 11 added");

    // Question 12: Testing Priorities
    var q12 = form.addCheckboxItem();
    q12.setTitle('Most important AI failures to test for');
    q12.setChoices([
      q12.createChoice('Unauthorized professional advice'),
      q12.createChoice('Industry-specific misinformation'),
      q12.createChoice('Privacy violations'),
      q12.createChoice('Inappropriate recommendations'),
      q12.createChoice('Safety restriction bypass'),
      q12.createChoice('Biased responses'),
      q12.createChoice('Regulatory violations')
    ]);
    q12.setRequired(true);
    Logger.log("Question 12 added");

    // Question 13: Additional Testing
    var q13 = form.addParagraphTextItem();
    q13.setTitle('Other AI safety concerns unique to your industry');
    q13.setHelpText('Any additional risks we should test for');
    Logger.log("Question 13 added");

    // Question 14: Role and Experience
    var q14 = form.addTextItem();
    q14.setTitle('Your role and years of experience');
    q14.setHelpText('e.g., Senior Financial Analyst, 8 years experience');
    q14.setRequired(true);
    Logger.log("Question 14 added");

    // Question 15: Email
    var q15 = form.addTextItem();
    q15.setTitle('Email (optional for follow-up)');
    q15.setHelpText('For clarification questions if needed');
    Logger.log("Question 15 added");

    // Question 16: Organization Type
    var q16 = form.addMultipleChoiceItem();
    q16.setTitle('Organization type');
    q16.setChoices([
      q16.createChoice('Large corporation (1000+ employees)'),
      q16.createChoice('Medium business (100-999 employees)'),
      q16.createChoice('Small business (10-99 employees)'),
      q16.createChoice('Government agency'),
      q16.createChoice('Healthcare facility'),
      q16.createChoice('Academic institution'),
      q16.createChoice('Other')
    ]);
    Logger.log("Question 16 added");

    // Question 17: Additional Comments
    var q17 = form.addParagraphTextItem();
    q17.setTitle('Additional comments about AI safety in your industry');
    q17.setHelpText('Any other insights or concerns');
    Logger.log("Question 17 added");

    // Set confirmation message
    form.setConfirmationMessage(
      'Thank you for completing the AI Safety Testing Configuration Form! ' +
      'We will review your responses and create custom AI safety tests for your industry.'
    );

    Logger.log('SUCCESS! Form created with 17 questions!');
    Logger.log('Published URL: ' + form.getPublishedUrl());
    Logger.log('Edit URL: ' + form.getEditUrl());

    return form;

  } catch (error) {
    Logger.log('Error at step: ' + error.toString());
    Logger.log('Error line: ' + error.lineNumber);
    throw error;
  }
}

/**
 * Even simpler version if the above fails
 */
function createSuperSimpleForm() {
  try {
    var form = FormApp.create('AI Safety Form - Simple');

    // Just 5 essential questions
    form.addTextItem().setTitle('What industry do you work in?').setRequired(true);
    form.addParagraphTextItem().setTitle('What are the biggest AI risks in your industry?').setRequired(true);
    form.addParagraphTextItem().setTitle('What regulations must AI comply with?').setRequired(true);
    form.addParagraphTextItem().setTitle('What language should raise red flags?').setRequired(true);
    form.addTextItem().setTitle('Your email for follow-up');

    Logger.log('Super simple form created: ' + form.getPublishedUrl());
    return form;

  } catch (error) {
    Logger.log('Even simple form failed: ' + error.toString());
  }
}

/**
 * Test individual form operations to find what's causing the error
 */
function testFormOperations() {
  try {
    Logger.log("Testing individual operations...");

    // Test 1: Create form
    var form = FormApp.create('Test Operations');
    Logger.log("✓ Form creation works");

    // Test 2: Set description
    form.setDescription('Test description');
    Logger.log("✓ Set description works");

    // Test 3: Set settings
    form.setCollectEmail(false);
    Logger.log("✓ Set collect email works");

    // Test 4: Add text item
    var textItem = form.addTextItem();
    textItem.setTitle('Test text question');
    Logger.log("✓ Add text item works");

    // Test 5: Add paragraph item
    var paragraphItem = form.addParagraphTextItem();
    paragraphItem.setTitle('Test paragraph question');
    Logger.log("✓ Add paragraph item works");

    // Test 6: Add multiple choice
    var mcItem = form.addMultipleChoiceItem();
    mcItem.setTitle('Test multiple choice');
    mcItem.setChoices([
      mcItem.createChoice('Option 1'),
      mcItem.createChoice('Option 2')
    ]);
    Logger.log("✓ Add multiple choice works");

    // Test 7: Set required
    textItem.setRequired(true);
    Logger.log("✓ Set required works");

    // Test 8: Set help text
    textItem.setHelpText('This is help text');
    Logger.log("✓ Set help text works");

    Logger.log("All basic operations work!");
    Logger.log("Form URL: " + form.getPublishedUrl());

    // Clean up
    DriveApp.getFileById(form.getId()).setTrashed(true);
    Logger.log("Test form deleted");

  } catch (error) {
    Logger.log("Error in test: " + error.toString());
    Logger.log("Failed at operation: " + error.message);
  }
}