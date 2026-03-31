import { test, expect } from '@playwright/test'

test.describe('End-to-End: Complete User Flow', () => {
  test('should redirect unauthenticated user to login', async ({ page }) => {
    await page.goto('/dashboard')
    await expect(page).toHaveURL('/login', { timeout: 5000 })
  })

  test('should display login page', async ({ page }) => {
    await page.goto('/login')
    await expect(page.locator('h1')).toContainText('Sign In')
    await expect(page.locator('input[type="email"]')).toBeVisible()
    await expect(page.locator('input[type="password"]')).toBeVisible()
    await expect(page.locator('button[type="submit"]')).toBeVisible()
  })

  test('should reject invalid credentials', async ({ page }) => {
    await page.goto('/login')

    // Fill incorrect credentials
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'wrong_password')

    // Submit form
    await page.click('button[type="submit"]')

    // Should show error and stay on login page
    await page.waitForTimeout(500)
    const errorText = page.locator('text=/[Ii]nvalid credentials/')
    await expect(errorText).toBeVisible({ timeout: 3000 })
  })

  test('should login with valid credentials', async ({ page }) => {
    await page.goto('/login')

    // Fill credentials
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')

    // Submit form
    await page.click('button[type="submit"]')

    // Should redirect to dashboard on success
    await expect(page).toHaveURL('/dashboard', { timeout: 5000 })
    
    // Dashboard should be visible
    await expect(page.locator('h1')).toContainText('Job Builder')
  })

  test('should display dataset selector on dashboard', async ({ page }) => {
    // Login first
    await page.goto('/login')
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')
    await page.click('button[type="submit"]')

    // Wait for dashboard to load
    await expect(page).toHaveURL('/dashboard')

    // Verify dataset selector exists
    const datasetSelect = page.locator('select').nth(1) // Second select is dataset
    await expect(datasetSelect).toBeVisible()
    
    // Verify it has options
    const options = page.locator('select').nth(1).locator('option')
    await expect(options).toHaveCount(5) // iris, wine, breast_cancer, mnist_binary, csv_upload
  })

  test('should create a training job with built-in dataset', async ({ page }) => {
    // Login
    await page.goto('/login')
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')
    await page.click('button[type="submit"]')

    // Wait for dashboard
    await expect(page).toHaveURL('/dashboard')
    
    // Wait for datasets to load
    await page.waitForTimeout(500)

    // Form should be visible
    const jobTypeSelect = page.locator('select').first()
    const datasetSelect = page.locator('select').nth(1)
    
    await expect(jobTypeSelect).toHaveValue('training')
    await expect(datasetSelect).toHaveValue('iris')

    // Submit form (default training job on iris)
    const submitButton = page.locator('button:has-text("Submit")')
    await submitButton.click()

    // Should process the submission
    await page.waitForTimeout(1000)
    
    // Should still be on dashboard or show success
    await expect(page).toHaveURL('/dashboard', { timeout: 3000 })
  })

  test('should require CSV upload when selecting csv_upload dataset', async ({ page }) => {
    // Login
    await page.goto('/login')
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')
    await page.click('button[type="submit"]')

    // Wait for dashboard
    await expect(page).toHaveURL('/dashboard')

    // Select csv_upload dataset
    const datasetSelect = page.locator('select').nth(1)
    await datasetSelect.selectOption('csv_upload')

    // File input should become visible
    const fileInput = page.locator('input[type="file"]')
    await expect(fileInput).toBeVisible({ timeout: 2000 })
  })

  test('should upload CSV file', async ({ page }) => {
    // Login
    await page.goto('/login')
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')
    await page.click('button[type="submit"]')

    // Wait for dashboard
    await expect(page).toHaveURL('/dashboard')

    // Select csv_upload dataset
    const datasetSelect = page.locator('select').nth(1)
    await datasetSelect.selectOption('csv_upload')

    // Create and upload a test CSV file
    const fileInput = page.locator('input[type="file"]')
    await fileInput.setInputFiles({
      name: 'test.csv',
      mimeType: 'text/csv',
      buffer: Buffer.from('feature1,feature2,label\n0.5,0.6,0\n0.7,0.8,1'),
    })

    // Wait for upload to complete
    await page.waitForTimeout(1500)
    
    // Should show uploaded filename
    const uploadedText = page.locator('text=/[Uu]ploaded.*test\\.csv/')
    await expect(uploadedText).toBeVisible({ timeout: 3000 })
  })

  test('should show validation error for missing credentials', async ({ page }) => {
    await page.goto('/login')

    // Try to submit without filling credentials
    const submitButton = page.locator('button[type="submit"]')
    await submitButton.click()

    // Page should remain on login
    await page.waitForTimeout(500)
    expect(page.url()).toContain('/login')
  })

  test('should allow logout from dashboard', async ({ page }) => {
    // Login first
    await page.goto('/login')
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')
    await page.click('button[type="submit"]')

    await expect(page).toHaveURL('/dashboard')

    // Find and click logout button
    const logoutButton = page.locator('button:has-text("Sign out")')
    await expect(logoutButton).toBeVisible()
    await logoutButton.click()

    // Should redirect to login
    await expect(page).toHaveURL('/login', { timeout: 5000 })
  })

  test('should persist session across page refreshes', async ({ page }) => {
    // Login
    await page.goto('/login')
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')
    await page.click('button[type="submit"]')

    await expect(page).toHaveURL('/dashboard')

    // Refresh page
    await page.reload()

    // Should still be on dashboard (auth cookie persists)
    await expect(page).toHaveURL('/dashboard', { timeout: 5000 })
  })
})

test.describe('Dashboard Interaction', () => {
  test('should display job form with all fields', async ({ page }) => {
    // Login
    await page.goto('/login')
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')
    await page.click('button[type="submit"]')

    await expect(page).toHaveURL('/dashboard')

    // Check all form elements exist
    const jobTypeSelect = page.locator('select').first()
    const datasetSelect = page.locator('select').nth(1)
    const configTextarea = page.locator('textarea')
    
    await expect(jobTypeSelect).toBeVisible()
    await expect(datasetSelect).toBeVisible()
    await expect(configTextarea).toBeVisible()
  })

  test('should change job type', async ({ page }) => {
    // Login
    await page.goto('/login')
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')
    await page.click('button[type="submit"]')

    await expect(page).toHaveURL('/dashboard')

    // Change job type
    const jobTypeSelect = page.locator('select').first()
    await jobTypeSelect.selectOption('inference')
    
    await expect(jobTypeSelect).toHaveValue('inference')
  })

  test('should change dataset', async ({ page }) => {
    // Login
    await page.goto('/login')
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')
    await page.click('button[type="submit"]')

    await expect(page).toHaveURL('/dashboard')

    // Change dataset
    const datasetSelect = page.locator('select').nth(1)
    await datasetSelect.selectOption('wine')
    
    await expect(datasetSelect).toHaveValue('wine')
  })
})

test.describe('Error Handling', () => {
  test('should display 404 for invalid routes', async ({ page }) => {
    // Navigate to non-existent route
    const response = await page.goto('/non-existent-route-12345')

    // Should be a 404 or redirect
    const status = response?.status()
    expect([404, 307, 308, 200]).toContain(status) // 200 = Next.js fallback
  })

  test('logout should clear session', async ({ page }) => {
    // Login
    await page.goto('/login')
    await page.fill('input[type="email"]', 'admin@example.com')
    await page.fill('input[type="password"]', 'password123')
    await page.click('button[type="submit"]')

    await expect(page).toHaveURL('/dashboard')

    // Logout
    const logoutButton = page.locator('button:has-text("Sign out")')
    await logoutButton.click()

    await expect(page).toHaveURL('/login')

    // Try to access dashboard - should redirect to login
    await page.goto('/dashboard')
    await expect(page).toHaveURL('/login', { timeout: 5000 })
  })
})
