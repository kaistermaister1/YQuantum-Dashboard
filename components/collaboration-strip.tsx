import Image from "next/image";
import qgarsMark from "@/public/qgars-filled.png";
import quantinuumMark from "@/subprojects/will/Travelers/docs/assets/logos/quantinuum-logo_horizontal_white.png";

export function CollaborationStrip() {
  return (
    <div className="collaboration-strip" aria-label="Project partners">
      <span className="collaboration-strip__logo-wrap collaboration-strip__logo-wrap--qgars">
        <Image src={qgarsMark} alt="Q-GARS" className="collaboration-strip__logo collaboration-strip__logo--qgars" priority />
      </span>
      <span className="collaboration-strip__divider" aria-hidden="true">
        x
      </span>
      <span className="collaboration-strip__logo-wrap">
        <img
          className="collaboration-strip__logo collaboration-strip__logo--ltm"
          src="/brand/ltm.svg"
          alt="LTM"
          width={180}
          height={47}
        />
      </span>
      <span className="collaboration-strip__divider" aria-hidden="true">
        x
      </span>
      <span className="collaboration-strip__logo-wrap collaboration-strip__logo-wrap--quantinuum">
        <Image
          src={quantinuumMark}
          alt="Quantinuum"
          className="collaboration-strip__logo collaboration-strip__logo--quantinuum"
        />
      </span>
      <span className="collaboration-strip__divider" aria-hidden="true">
        x
      </span>
      <span className="collaboration-strip__sponsor-lockup">
        <img
          className="collaboration-strip__logo collaboration-strip__logo--travelers"
          src="/brand/travelers.svg"
          alt="Travelers"
          width={53}
          height={48}
        />
      </span>
    </div>
  );
}
