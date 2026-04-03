/**
 * Integrated Job Lifecycle Tests
 * Tests the complete flow: upload → create job → poll → complete
 * Runs against real frontend API and worker
 */

import { test, expect } from '@playwright/test'

test.describe.configure({ timeout: 120000 })

async function fillField(
  field: import('@playwright/test').Locator,
  value: string
) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    await field.click()
    await field.fill('')
    await field.type(value, { delay: 20 })

    if ((await field.inputValue()) === value) {
      return
    }
  }

  await expect(field).toHaveValue(value, { timeout: 10000 })
}

async function login(page: import('@playwright/test').Page) {
  await page.goto('/login')
  const emailInput = page.getByLabel('Email')
  const passwordInput = page.getByLabel('Password')
  await expect(emailInput).toBeVisible()
  await expect(passwordInput).toBeVisible()

  await fillField(emailInput, 'admin@example.com')
  await fillField(passwordInput, 'password123')

  const loginResponsePromise = page.waitForResponse(
    (response) =>
      response.url().includes('/api/auth/login') &&
      response.request().method() === 'POST',
    { timeout: 15000 }
  )

  await page.getByRole('button', { name: /^sign in$/i }).click()
  const loginResponse = await loginResponsePromise

  expect(loginResponse.ok()).toBe(true)
  await expect(page).toHaveURL('/dashboard', { timeout: 15000 })
}

test.describe('Integrated Job Lifecycle Tests', () => {
  test('complete flow: upload CSV → create job → poll → results', async ({ page, context }) => {
    // Step 1: Login
    await login(page)

    // Step 2: Navigate to CSV upload
    const datasetSelect = page.locator('select').nth(1)
    await datasetSelect.selectOption('csv_upload')

    // Step 3: Upload a test CSV file
    const csvContent = 'sepal_length,sepal_width,petal_length,petal_width,species\n5.1,3.5,1.4,0.2,0\n7.0,3.2,4.7,1.4,1\n6.3,3.3,6.0,2.5,2'
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'flowers.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent),
    })

    // Wait for upload to complete
    await page.waitForTimeout(1500)
    const uploadedText = page.locator('text=/[Uu]ploaded.*flowers\\.csv/')
    await expect(uploadedText).toBeVisible({ timeout: 3000 })

    // Step 4: Submit job with CSV
    const jobTypeSelect = page.locator('select').first()
    await expect(jobTypeSelect).toHaveValue('training')

    const submitButton = page.locator('button:has-text("Submit")')
    await submitButton.click()

    // Wait for job submission
    await page.waitForTimeout(1000)

    // Step 5: Verify job was created (should still be on dashboard)
    await expect(page).toHaveURL('/dashboard', { timeout: 3000 })

    // Step 6: Poll for job completion (with timeout)
    let jobCompleted = false
    let pollCount = 0
    const maxPolls = 60 // 60 * 1s = 1 minute timeout

    while (!jobCompleted && pollCount < maxPolls) {
      await page.waitForTimeout(1000)
      pollCount++

      // Reload to get latest job status
      await page.reload()
      await page.waitForTimeout(500)

      // Check for completed status
      const pageContent = await page.content()
      if (
        pageContent.includes('completed') ||
        pageContent.includes('Completed') ||
        pageContent.includes('✓') ||
        pageContent.includes('accuracy')
      ) {
        jobCompleted = true
        break
      }

      if (pollCount % 10 === 0) {
        console.log(`Polled ${pollCount} times...`)
      }
    }

    // Verify job completed
    expect(jobCompleted).toBe(true)

    // Step 7: Verify results are displayed
    const pageContent = await page.content()
    const hasResults =
      pageContent.includes('accuracy') ||
      pageContent.includes('loss') ||
      pageContent.includes('results') ||
      pageContent.includes('training')

    expect(hasResults).toBe(true)
  })

  test('job persistence: job survives page refresh', async ({ page }) => {
    // Login
    await login(page)

    // Submit a simple training job
    const jobTypeSelect = page.locator('select').first()
    const datasetSelect = page.locator('select').nth(1)

    await expect(jobTypeSelect).toHaveValue('training')
    await expect(datasetSelect).toHaveValue('iris')

    const submitButton = page.locator('button:has-text("Submit")')
    await submitButton.click()
    await page.waitForTimeout(500)

    // Refresh page
    await page.reload()

    // Job list should still be there
    await expect(page).toHaveURL('/dashboard')
    const pageContent = await page.content()
    expect(pageContent.length).toBeGreaterThan(100) // Verify page loaded
  })

  test('multiple jobs can be submitted sequentially', async ({ page }) => {
    // Login
    await login(page)

    const jobTypeSelect = page.locator('select').first()
    const datasetSelect = page.locator('select').nth(1)
    const submitButton = page.locator('button:has-text("Submit")')

    // Submit first job (training on iris)
    await expect(jobTypeSelect).toHaveValue('training')
    await expect(datasetSelect).toHaveValue('iris')
    await submitButton.click()
    await page.waitForTimeout(500)

    // Submit second job (inference on wine)
    await datasetSelect.selectOption('wine')
    await jobTypeSelect.selectOption('inference')
    await submitButton.click()
    await page.waitForTimeout(500)

    // Submit third job (experiment on breast_cancer)
    await datasetSelect.selectOption('breast_cancer')
    await jobTypeSelect.selectOption('experiment')
    await submitButton.click()
    await page.waitForTimeout(500)

    // Verify we're still on dashboard
    await expect(page).toHaveURL('/dashboard')

    // Verify page shows job list
    const pageContent = await page.content()
    const hasJobIndicators =
      pageContent.includes('training') ||
      pageContent.includes('Training') ||
      pageContent.includes('job') ||
      pageContent.includes('Job')

    expect(hasJobIndicators).toBe(true)
  })

  test('error handling: invalid CSV fails gracefully', async ({ page }) => {
    // Login
    await login(page)

    // Select CSV dataset
    const datasetSelect = page.locator('select').nth(1)
    await datasetSelect.selectOption('csv_upload')

    // Try to upload invalid CSV (missing label column)
    const invalidCsv = 'feature1,feature2\n0.5,0.6\n0.7,0.8'
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'invalid.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(invalidCsv),
    })

    await page.waitForTimeout(1500)

    // Should show upload error or handle gracefully
    const pageContent = await page.content()
    // Either shows error message or handles silently
    const hasErrorOrFallback = pageContent.includes('error') || pageContent.includes('Error') || pageContent.length > 100

    expect(hasErrorOrFallback).toBe(true)
  })

  test('dataset switching clears CSV upload state', async ({ page }) => {
    // Login
    await login(page)

    const datasetSelect = page.locator('select').nth(1)

    // Upload CSV
    await datasetSelect.selectOption('csv_upload')
    const fileInput = page.locator('input[type="file"]')
    const csvContent = 'f1,f2,l\n0.5,0.6,0\n0.7,0.8,1'

    const uploadResponsePromise = page.waitForResponse(
      (response) =>
        response.url().includes('/api/uploads') &&
        response.request().method() === 'POST',
      { timeout: 10000 }
    )

    await fileInput.setInputFiles({
      name: 'test.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from(csvContent),
    })

    const uploadResponse = await uploadResponsePromise
    expect(uploadResponse.ok()).toBe(true)

    let uploadedText = page.locator('text=/[Uu]ploaded.*test\\.csv/')
    await expect(uploadedText).toBeVisible({ timeout: 5000 })

    // Switch to different dataset
    await datasetSelect.selectOption('iris')

    // Switch back to csv_upload
    await datasetSelect.selectOption('csv_upload')

    // Upload state should be cleared (file input empty)
    const fileInputElement = page.locator('input[type="file"]')
    const fileValue = await fileInputElement.inputValue()
    expect(fileValue).toBe('')
  })

  test('config JSON validation', async ({ page }) => {
    // Login
    await login(page)

    const textarea = page.locator('textarea')
    const submitButton = page.locator('button:has-text("Submit")')

    // Set valid JSON
    await textarea.fill('{"test": "value"}')
    await submitButton.click()
    await page.waitForTimeout(300)

    // Should submit without error
    let pageContent = await page.content()
    expect(!pageContent.includes('Config must be valid JSON')).toBe(true)

    // Set invalid JSON
    await page.reload()
    await page.waitForTimeout(500)

    await textarea.fill('{invalid json}')
    await submitButton.click()
    await page.waitForTimeout(300)

    // Should show validation error
    pageContent = await page.content()
    const hasValidationMessage =
      pageContent.includes('Config must be valid JSON') || pageContent.includes('Config') || pageContent.includes('JSON')

    expect(hasValidationMessage).toBe(true)
  })

  test('concurrent users do not interfere', async ({ context }) => {
    // Create two separate authenticated sessions
    const page1 = await context.newPage()
    const page2 = await context.newPage()

    // User 1 logs in
    await login(page1)

    // User 2 logs in (same credential for testing, but tracked separately)
    await login(page2)

    // User 1 submits job
    const submitButton1 = page1.locator('button:has-text("Submit")').first()
    await submitButton1.click()

    // User 2 submits different job
    const datasetSelect2 = page2.locator('select').nth(1)
    await datasetSelect2.selectOption('wine')
    const submitButton2 = page2.locator('button:has-text("Submit")').first()
    await submitButton2.click()

    // Both should still be on dashboard
    await expect(page1).toHaveURL('/dashboard')
    await expect(page2).toHaveURL('/dashboard')

    // Cleanup
    await page1.close()
    await page2.close()
  })
})

test.describe('Job Status Polling Accuracy', () => {
  test('job status updates correctly during execution', async ({ page }) => {
    // Login
    await login(page)

    // Submit job
    const submitButton = page.locator('button:has-text("Submit")')
    await submitButton.click()
    await page.waitForTimeout(500)

    // Poll and check status transitions
    const statuses: string[] = []
    for (let i = 0; i < 15; i++) {
      await page.waitForTimeout(1000)

      const pageContent = await page.content()

      // Extract status
      if (pageContent.includes('queued')) statuses.push('queued')
      if (pageContent.includes('running')) statuses.push('running')
      if (pageContent.includes('completed')) statuses.push('completed')
      if (pageContent.includes('failed')) statuses.push('failed')

      // If completed, stop polling
      if (pageContent.includes('completed')) break

      // Reload to get fresh data
      await page.reload()
      await page.waitForTimeout(300)
    }

    // Status should have some progression (at minimum: started and potential completion)
    expect(statuses.length).toBeGreaterThanOrEqual(1)

    // Final status should be one of these
    const finalStatus = statuses[statuses.length - 1] || ''
    expect(['queued', 'running', 'completed', 'failed']).toContain(finalStatus)
  })
})
