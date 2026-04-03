import { z } from "zod";

export const DatasetIdSchema = z.enum([
  "iris",
  "breast_cancer",
  "wine",
  "mnist_binary",
  "csv_upload",
]);

export const JobTypeSchema = z.enum(["experiment", "training", "inference"]);

export const JobStatusSchema = z.enum([
  "queued",
  "running",
  "completed",
  "failed",
]);

export const CreateJobInputSchema = z.object({
  jobType: JobTypeSchema,
  datasetId: DatasetIdSchema,
  config: z.record(z.string(), z.unknown()).default({}),
  csvBlobUrl: z.string().url().optional(),
});

export const JobSummarySchema = z.object({
  id: z.string().min(8),
  jobType: JobTypeSchema,
  datasetId: DatasetIdSchema,
  status: JobStatusSchema,
  createdAt: z.string(),
  updatedAt: z.string(),
  message: z.string().optional(),
  result: z.record(z.string(), z.unknown()).optional(),
});

export const DatasetSchema = z.object({
  id: DatasetIdSchema,
  label: z.string(),
  nClasses: z.number().int().positive(),
  defaultQubits: z.number().int().positive(),
});

export type DatasetId = z.infer<typeof DatasetIdSchema>;
export type JobType = z.infer<typeof JobTypeSchema>;
export type JobStatus = z.infer<typeof JobStatusSchema>;
export type CreateJobInput = z.infer<typeof CreateJobInputSchema>;
export type JobSummary = z.infer<typeof JobSummarySchema>;
export type DatasetInfo = z.infer<typeof DatasetSchema>;
