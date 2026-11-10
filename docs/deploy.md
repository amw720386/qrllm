# Deploy QRLLM

Publish the contents of `dist/` at the HTTPS origin
`https://qrllm.ahamedwajibu.com/`. The site consists of HTML, CSS and two
JavaScript modules. The static host must serve `.mjs` as JavaScript.

## Cloudflare Pages

1. Create a Pages Direct Upload project and upload the `dist/` directory.
   For a CLI deployment with Wrangler installed and authenticated, run
   `wrangler pages deploy dist --project-name qrllm`. The supplied
   `wrangler.toml` and `dist/_headers` set the output directory and response
   headers. [Cloudflare Pages Direct Upload](https://developers.cloudflare.com/pages/get-started/direct-upload/)
2. Add `qrllm.ahamedwajibu.com` as a custom domain in the Pages project and
   follow the host's DNS instructions. Wait for the HTTPS certificate.
   [Cloudflare Pages custom domains](https://developers.cloudflare.com/pages/configuration/custom-domains/)
3. Visit the bare origin, then open [the full QR URL](../demo/open.md).
   The bare origin should display “what are you doing here?” The model link
   should generate a name. The Network panel should show just the static
   application and zero requests while generating.

`demo/`, `data/`, `work/`, and `tests/` belong outside the deployment.
Hosting `dist/` as the site root preserves that separation. Avoid redirects
and link wrappers that discard the fragment. If the host name changes, rerun
`python tools/export.py --origin https://new.example/` and verify the result.

## Before sharing the PNG

- Display the full image on a monitor and scan it with the default iPhone and
  Android cameras. Confirm the browser shows a generated name.
- Save the image to each phone and open the detected link using Photos or
  Google Lens. Check that the full model fragment survives.
- After you upload the PNG to X, repeat the scan against the image X actually
  serves. Try the expanded image as well as the feed thumbnail.
- Share the [complete model link](../demo/open.md) as a same-phone fallback.
  If a platform wrapper strips the fragment, copy the full URL from
  [model-url.txt](../demo/model-url.txt) into the browser's address bar.
