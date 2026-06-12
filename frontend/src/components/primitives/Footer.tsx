import Link from "next/link";

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-[var(--color-border)] bg-[var(--color-surface-sunken)]">
      <div className="mx-auto grid max-w-7xl gap-10 px-[var(--spacing-page-x)] py-12 md:grid-cols-4 lg:px-[var(--spacing-page-x-lg)]">
        <div>
          <p className="heading-h3 text-[var(--color-text)]">SignLearn</p>
          <p className="mt-2 text-sm text-[var(--color-text-muted)] max-w-xs leading-relaxed">
            Real-time American Sign Language ↔ English in your browser. Open source, on-device, research-driven.
          </p>
        </div>

        <FooterCol heading="Product">
          <FooterLink href="/practice">Practice</FooterLink>
          <FooterLink href="/learn">Learn</FooterLink>
          <FooterLink href="/analytics">Analytics</FooterLink>
        </FooterCol>

        <FooterCol heading="Community">
          <FooterLink href="/research">Send feedback</FooterLink>
          <FooterLink href="/research?tab=correction">Report a wrong prediction</FooterLink>
          <FooterLink href="/research?tab=study">Join a study</FooterLink>
        </FooterCol>

        <FooterCol heading="About">
          <FooterLink href="/privacy">Privacy & data</FooterLink>
          <FooterLink href="/accessibility">Accessibility statement</FooterLink>
          <FooterLink href="https://github.com/" external>GitHub</FooterLink>
        </FooterCol>
      </div>

      <div className="border-t border-[var(--color-border)]">
        <p className="mx-auto max-w-7xl px-[var(--spacing-page-x)] py-5 text-xs text-[var(--color-text-faint)] lg:px-[var(--spacing-page-x-lg)]">
          © {new Date().getFullYear()} SignLearn. Research project, not a medical or clinical tool. Built with input from Deaf community contributors.
        </p>
      </div>
    </footer>
  );
}

function FooterCol({ heading, children }: { heading: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="eyebrow">{heading}</p>
      <ul className="mt-3 space-y-2 text-sm">{children}</ul>
    </div>
  );
}

function FooterLink({ href, external, children }: { href: string; external?: boolean; children: React.ReactNode }) {
  if (external) {
    return (
      <li>
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
        >
          {children}
        </a>
      </li>
    );
  }
  return (
    <li>
      <Link href={href} className="text-[var(--color-text-muted)] hover:text-[var(--color-text)]">
        {children}
      </Link>
    </li>
  );
}
