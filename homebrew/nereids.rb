# Template only — the canonical cask lives at ornlneutronimaging/homebrew-nereids.
# CI (publish.yml update-homebrew job) substitutes version + sha256 in the tap repo.
cask "nereids" do
  version "0.1.5"
  sha256 :no_check

  url "https://github.com/ornlneutronimaging/NEREIDS/releases/download/v#{version}/nereids-#{version}-macos-arm64.dmg"
  name "NEREIDS"
  desc "Neutron resonance imaging desktop application"
  homepage "https://github.com/ornlneutronimaging/NEREIDS"

  depends_on macos: ">= :big_sur"
  depends_on arch: :arm64

  app "NEREIDS.app"

  postflight do
    system_command "/usr/bin/xattr",
                   args: ["-cr", "#{appdir}/NEREIDS.app"]
  end

  zap trash: [
    "~/Library/Preferences/gov.ornl.nereids.plist",
    "~/Library/Saved Application State/gov.ornl.nereids.savedState",
  ]
end
