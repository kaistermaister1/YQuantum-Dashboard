import Image from "next/image";
import qgarsMark from "@/qgars.png";

export function CollaborationStrip() {
  return (
    <header className="collaboration-strip" aria-label="Q-gars in collaboration with challenge sponsors">
      <div className="collaboration-strip__cluster collaboration-strip__cluster--team">
        <Image
          src={qgarsMark}
          alt="Q-gars"
          width={520}
          height={154}
          className="collaboration-strip__qgars"
          priority
        />
      </div>

      <span className="collaboration-strip__times" aria-hidden="true">
        ×
      </span>

      <div className="collaboration-strip__cluster collaboration-strip__cluster--sponsors">
        <span className="collaboration-strip__sponsor-lockup">
          <img
            className="collaboration-strip__sponsor-logo collaboration-strip__sponsor-logo--travelers"
            src="/brand/travelers.svg"
            alt="Travelers"
            width={53}
            height={48}
          />
        </span>
        <span className="collaboration-strip__sponsor-lockup">
          <img
            className="collaboration-strip__sponsor-logo collaboration-strip__sponsor-logo--ltm"
            src="/brand/ltm.svg"
            alt="LTM"
            width={180}
            height={47}
          />
        </span>
      </div>
    </header>
  );
}
