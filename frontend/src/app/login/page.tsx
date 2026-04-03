import LoginClientPage from "./LoginClientPage";

type LoginPageProps = {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const resolvedSearchParams = await searchParams;
  const rawError = resolvedSearchParams.error;
  const oauthErrorCode = Array.isArray(rawError) ? rawError[0] ?? null : rawError ?? null;

  return <LoginClientPage oauthErrorCode={oauthErrorCode} />;
}
